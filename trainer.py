import json
import logging
import math
import os
import random
import re
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from sklearn.metrics import roc_curve, auc
import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from tqdm.auto import tqdm, trange

from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, PredictionOutput, TrainOutput
from transformers.training_args import TrainingArguments

def is_tpu_available():
    return False

try:
    from apex import amp

    _has_apex = True
except ImportError:
    _has_apex = False

TEST_ID_STRS = [
    'chronic_PTSD_0',
    'type_2_diabtes_0',
    'hyperlipidemia_0',
    'loin_pain_0',
    'low_back_pain_0',
    'PTSD_0',
    'obstructive_sleep_apnea_hypopnea_0',
    'mental_depression_0',
    'chronic_obstructive_airway_disease_0',
    'sensorineural_hearing_loss_0',
    'gastroesophagel_reflux_disease_without_esophagitis_0',
    'gastroesophagel_reflux_disease_0',
    'coronary_arteriosclerosis_0',
    'arteriosclerotic_heart_disease_0',
    'chronic_PTSD_3',
    'type_2_diabtes_3',
    'hyperlipidemia_3',
    'loin_pain_3',
    'low_back_pain_3',
    'PTSD_3',
    'obstructive_sleep_apnea_hypopnea_3',
    'mental_depression_3',
    'chronic_obstructive_airway_disease_3',
    'sensorineural_hearing_loss_3',
    'gastroesophagel_reflux_disease_without_esophagitis_3',
    'gastroesophagel_reflux_disease_3',
    'coronary_arteriosclerosis_3',
    'arteriosclerotic_heart_disease_3',
    'chronic_PTSD_6',
    'type_2_diabtes_6',
    'hyperlipidemia_6',
    'loin_pain_6',
    'low_back_pain_6',
    'PTSD_6',
    'obstructive_sleep_apnea_hypopnea_6',
    'mental_depression_6',
    'chronic_obstructive_airway_disease_6',
    'sensorineural_hearing_loss_6',
    'gastroesophagel_reflux_disease_without_esophagitis_6',
    'gastroesophagel_reflux_disease_6',
    'coronary_arteriosclerosis_6',
    'arteriosclerotic_heart_disease_6',
    'least_happen'
]

def is_apex_available():
    return _has_apex


if is_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_tensorboard_available():
    return _has_tensorboard


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except ImportError:
    _has_wandb = False


def is_wandb_available():
    return _has_wandb


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: Dataset):
    if xm.xrt_world_size() <= 1:
        return RandomSampler(dataset)
    return DistributedSampler(dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())


class Trainer:
    """
    Trainer is a simple but feature-complete training and eval loop for PyTorch,
    optimized for Transformers.
    """

    model: PreTrainedModel
    args: TrainingArguments
    data_collator: DataCollator
    train_dataset: Optional[Dataset]
    eval_dataset: Optional[Dataset]
    test_datasets: Optional[List[Dataset]]
    test_collator: Optional[DataCollator]
    compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None
    prediction_loss_only: bool
    tb_writer: Optional["SummaryWriter"] = None
    optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None
    global_step: Optional[int] = None
    epoch: Optional[float] = None

    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        test_datasets: Optional[List[Dataset]] = None,
        test_collator: Optional[DataCollator] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        prediction_loss_only=False,
        tb_writer: Optional["SummaryWriter"] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
    ):
        """
        Trainer is a simple but feature-complete training and eval loop for PyTorch,
        optimized for Transformers.

        Args:
            prediction_loss_only:
                (Optional) in evaluation and prediction, only return the loss
        """
        self.model = model.to(args.device)
        self.args = args
        if data_collator is not None:
            self.data_collator = data_collator
        else:
            self.data_collator = DefaultDataCollator()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_datasets = test_datasets
        if test_collator is not None:
            self.test_collator = test_collator
        else:
            self.test_collator = DefaultDataCollator()
        self.compute_metrics = compute_metrics
        self.prediction_loss_only = prediction_loss_only
        self.optimizers = optimizers
        if tb_writer is not None:
            self.tb_writer = tb_writer
        elif is_tensorboard_available() and self.is_world_master():
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)
        if not is_tensorboard_available():
            logger.warning(
                "You are instantiating a Trainer but Tensorboard is not installed. You should consider installing it."
            )
        if is_wandb_available():
            self._setup_wandb()
        else:
            logger.info(
                "You are instantiating a Trainer but W&B is not installed. To use wandb logging, "
                "run `pip install wandb; wandb login` see https://docs.wandb.com/huggingface."
            )
        set_seed(self.args.seed)
        # Create output directory if needed
        if self.is_world_master():
            os.makedirs(self.args.output_dir, exist_ok=True)
        if is_tpu_available():
            # Set an xla_device flag on the model's config.
            # We'll find a more elegant and not need to do this in the future.
            self.model.config.xla_device = True

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_tpu_available():
            train_sampler = get_tpu_sampler(self.train_dataset)
        else:
            train_sampler = (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(eval_dataset)
        else:
            sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_test_dataloaders(self, test_datasets: Optional[List[Dataset]] = None) -> List[DataLoader]:
        if test_datasets is None and self.test_datasets is None:
            raise ValueError("Trainer: evaluation requires an test_datasets.")

        test_datasets = test_datasets if test_datasets is not None else self.test_datasets

        data_loaders = []
        for test_dataset in test_datasets:
            if is_tpu_available():
                sampler = SequentialDistributedSampler(
                    test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
                )
            elif self.args.local_rank != -1:
                sampler = SequentialDistributedSampler(test_dataset)
            else:
                sampler = SequentialSampler(test_dataset)

            data_loader = DataLoader(
                test_dataset,
                sampler=sampler,
                batch_size=self.args.eval_batch_size,
                collate_fn=self.test_collator.collate_batch,
            )

            data_loaders.append(data_loader)

        return data_loaders

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        # We use the same batch_size as for eval.
        if is_tpu_available():
            sampler = SequentialDistributedSampler(
                test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal()
            )
        elif self.args.local_rank != -1:
            sampler = SequentialDistributedSampler(test_dataset)
        else:
            sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator.collate_batch,
        )

        return data_loader

    def get_optimizers(
        self, num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well.
        If you want to use something else, you can pass a tuple in the Trainer's init,
        or override this method in a subclass.
        """
        if self.optimizers is not None:
            return self.optimizers
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def _setup_wandb(self):
        """
        Setup the optional Weights & Biases (`wandb`) integration.

        One can override this method to customize the setup if needed.  Find more information at https://docs.wandb.com/huggingface
        You can also override the following environment variables:

        Environment:
            WANDB_WATCH:
                (Optional, ["gradients", "all", "false"]) "gradients" by default, set to "false" to disable gradient logging
                or "all" to log gradients and parameters
            WANDB_PROJECT:
                (Optional): str - "huggingface" by default, set this to a custom string to store results in a different project
            WANDB_DISABLED:
                (Optional): boolean - defaults to false, set to "true" to disable wandb entirely
        """
        logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
        wandb.init(project=os.getenv("WANDB_PROJECT", "huggingface"), config=vars(self.args))
        # keep track of model topology and gradients
        if os.getenv("WANDB_WATCH") != "false":
            wandb.watch(
                self.model, log=os.getenv("WANDB_WATCH", "gradients"), log_freq=max(100, self.args.logging_steps)
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get num of examples from a DataLoader, by accessing its Dataset.
        """
        return len(dataloader.dataset)

    def train(self, model_path: Optional[str] = None):
        """
        Main training entry point.

        Args:
            model_path:
                (Optional) Local path to model if model to train has been instantiated from a local path
                If present, we will try reloading the optimizer/scheduler states from there.
        """
        train_dataloader = self.get_train_dataloader()
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs

        optimizer, scheduler = self.get_optimizers(num_training_steps=t_total)

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model
        if self.args.fp16:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        if self.tb_writer is not None:
            self.tb_writer.add_text("args", self.args.to_json_string())
            self.tb_writer.add_hparams(self.args.to_sanitized_dict(), metric_dict={})

        # Train!
        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = 0.0
        logging_loss = 0.0
        model.zero_grad()
        train_iterator = trange(
            epochs_trained, int(num_train_epochs), desc="Epoch", disable=not self.is_local_master(), mininterval=3600
        )
        for epoch in train_iterator:
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)


            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=not self.is_local_master(), mininterval=600)

            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                tr_loss += self._training_step(model, inputs, optimizer)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        if self.global_step == 1 and self.args.logging_first_step:
                            logs["loss"] = (tr_loss - logging_loss)
                        else:
                            logs["loss"] = (tr_loss - logging_loss) / self.args.logging_steps
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss = tr_loss

                        self._log(logs)

                        # if self.args.evaluate_during_training:
                        self.evaluate()

                    if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                        # In all cases (even distributed/parallel), self.model is always a reference
                        # to the model we want to save.
                        if hasattr(model, "module"):
                            assert model.module is self.model
                        else:
                            assert model is self.model
                        # Save model checkpoint
                        output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.global_step}")

                        self.save_model(output_dir)

                        if self.is_world_master():
                            self._rotate_checkpoints()

                        if is_tpu_available():
                            xm.rendezvous("saving_optimizer_states")
                            xm.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            xm.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        elif self.is_world_master():
                            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                train_iterator.close()
                break
            if self.args.tpu_metrics_debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.tb_writer:
            self.tb_writer.close()

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, None)

    def _log(self, logs: Dict[str, float], iterator: Optional[tqdm] = None) -> None:
        if self.epoch is not None:
            logs["epoch"] = self.epoch
        if self.tb_writer:
            for k, v in logs.items():
                self.tb_writer.add_scalar(k, v, self.global_step)
        if is_wandb_available():
            wandb.log(logs, step=self.global_step)
        output = json.dumps({**logs, **{"step": self.global_step}})
        if iterator is not None:
            iterator.write(output)
        else:
            print(output)

    def _training_step(
        self, model: nn.Module, inputs: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        for k, v in inputs.items():
            inputs[k] = v.to(self.args.device)

        outputs = model(**inputs)
        loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        return loss.item()

    def is_local_master(self) -> bool:
        if is_tpu_available():
            return xm.is_master_ordinal(local=True)
        else:
            return self.args.local_rank in [-1, 0]

    def is_world_master(self) -> bool:
        """
        This will be True only in one process, even in distributed mode,
        even when training on multiple machines.
        """
        if is_tpu_available():
            return xm.is_master_ordinal(local=False)
        else:
            return self.args.local_rank == -1 or torch.distributed.get_rank() == 0

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().

        Will only save from the world_master process (unless in TPUs).
        """

        if is_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_master():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")

        xm.rendezvous("saving_checkpoint")
        self.model.save_pretrained(output_dir)

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, PreTrainedModel):
            raise ValueError("Trainer.model appears to not be a PreTrainedModel")
        self.model.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
        return checkpoints_sorted

    def _rotate_checkpoints(self, use_mtime=False) -> None:
        if self.args.save_total_limit is None or self.args.save_total_limit <= 0:
            return

        # Check if we should delete older checkpoint(s)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=use_mtime)
        if len(checkpoints_sorted) <= self.args.save_total_limit:
            return

        number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - self.args.save_total_limit)
        checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
        for checkpoint in checkpoints_to_be_deleted:
            logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
            shutil.rmtree(checkpoint)

    def evaluate(
        self, eval_dataset: Optional[Dataset] = None, prediction_loss_only: Optional[bool] = None,
    ) -> Dict[str, float]:
        """
        Run evaluation and return metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent.

        Args:
            eval_dataset: (Optional) Pass a dataset if you wish to override
            the one on the instance.
        Returns:
            A dict containing:
                - the eval loss
                - the potential metrics computed from the predictions
        """
        if self.test_datasets is not None:
            test_dataloaders = self.get_test_dataloaders()
            for idx, test_dataloader in enumerate(test_dataloaders):
                output = self._prediction_loop(test_dataloader, description="%02d_%s"%(idx+1,TEST_ID_STRS[idx]), is_test=True) 
                self._log(output.metrics)


        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self._prediction_loop(eval_dataloader, description="00_valid")
        self._log(output.metrics)

        if self.args.tpu_metrics_debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output.metrics

    def predict(self, test_dataset: Dataset) -> PredictionOutput:
        """
        Run prediction and return predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels.
        In that case, this method will also return metrics, like in evaluate().
        """
        test_dataloader = self.get_test_dataloader(test_dataset)

        return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None, is_test = False
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        save_act_predict_result= True
        save_count=0

        def calc_overlap_todel(logits:torch.Tensor, labels:torch.Tensor, device=None) -> torch.Tensor:
            num_batch = logits.shape[0] #(batch, toks, vocab_size)
            assert num_batch == labels.shape[0] # check same number of batch (batch, toks)
            result = 0.0
            for i in range(num_batch):
                label = labels[i][labels[i]!=-100]
                logit = logits[i][labels[i]!=-100]
                pred = torch.argmax(logit.detach(), axis=1)

                label = label.detach().cpu().numpy().tolist()
                label = set(label)
                pred = pred.detach().cpu().numpy().tolist()
                pred = set(pred)
                overlap = pred.intersection(label)
                joint = pred.union(label)
                # result += len(overlap)/float(len(label))
                result += len(overlap)/float(len(joint)) #TODO: change to joint
            return torch.tensor(result, dtype=torch.float, device=device) 

        def calc_overlap_bart_todel(logits:torch.Tensor, labels:torch.Tensor, device=None) -> torch.Tensor:
            num_batch = logits.shape[0] #(batch, toks, vocab_size)
            assert num_batch == labels.shape[0] # check same number of batch (batch, toks)
            result = 0.0
            pred_len = []
            actu_len = []

            for i in range(num_batch):
                label = labels[i][labels[i]!=-100]
                logit = logits[i][labels[i]!=-100]
                pred = torch.argmax(logit.detach(), axis=1)


                tmp = label[0]
                assert tmp>200 and tmp <300
                actu_len.append(tmp.item())
                label = label[1:]

                tmp = pred[0]
                pred_len.append(tmp.item())
                pred = pred[1:]

                label = label.detach().cpu().numpy().tolist()
                label = set(label)
                pred  = pred.detach().cpu().numpy().tolist()
                pred  = set(pred)
                overlap = pred.intersection(label)
                joint = pred.union(label)
                # result += len(overlap)/float(len(label))
                result += len(overlap)/float(len(joint)) #TODO: change to joint
                # roc_auc = auc_metrics(yhat_raw, y, ymic)
            pred_len = np.array(pred_len)
            actu_len = np.array(actu_len)
            return torch.tensor(result, dtype=torch.float, device=device), torch.tensor(pred_len==actu_len, dtype=torch.float, device=device).sum()

        def auc_metrics(yhat_raw, y, ymic):
            if yhat_raw.shape[0] <= 1:
                return
            fpr = {}
            tpr = {}
            roc_auc = {}
            #get AUC for each label individually
            relevant_labels = []
            auc_labels = {}
            for i in range(y.shape[1]):
                #only if there are true positives for this label
                if y[:,i].sum() > 0:
                    fpr[i], tpr[i], _ = roc_curve(y[:,i], yhat_raw[:,i])
                    if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                        auc_score = auc(fpr[i], tpr[i])
                        if not np.isnan(auc_score): 
                            auc_labels["auc_%d" % i] = auc_score
                            relevant_labels.append(i)

            #macro-AUC: just average the auc scores
            aucs = []
            for i in relevant_labels:
                aucs.append(auc_labels['auc_%d' % i])
            roc_auc['auc_macro'] = np.mean(aucs)

            #micro-AUC: just look at each individual prediction
            yhatmic = yhat_raw.ravel()
            fpr["micro"], tpr["micro"], _ = roc_curve(ymic, yhatmic) 
            roc_auc["auc_micro"] = auc(fpr["micro"], tpr["micro"])

            return roc_auc

        def calc_overlap(logits:torch.Tensor, labels:torch.Tensor, device=None) -> torch.Tensor:
            num_batch = logits.shape[0] #(batch, toks, vocab_size)
            vocab_size = logits.shape[2] #(batch, toks, vocab_size)
            assert num_batch == labels.shape[0] # check same number of batch (batch, toks)
            result = 0.0
            for i in range(num_batch):
                label = labels[i][labels[i]!=-100]
                logit = logits[i][labels[i]!=-100]
                pred = torch.argmax(logit.detach(), axis=1)

                label = label.detach().cpu().numpy().tolist()
                label = set(label)
                pred = pred.detach().cpu().numpy().tolist()
                pred = set(pred)
                overlap = pred.intersection(label)
                joint = pred.union(label)
                # result += len(overlap)/float(len(label))
                result += len(overlap)/float(len(joint)) #TODO: change to joint
            return torch.tensor(result, dtype=torch.float, device=device) 

        def calc_overlap_bart(logits:torch.Tensor, labels:torch.Tensor, device=None) -> torch.Tensor:
            num_batch = logits.shape[0] #(batch, toks, vocab_size)
            vocab_size = logits.shape[2] #(batch, toks, vocab_size)
            assert num_batch == labels.shape[0] # check same number of batch (batch, toks)
            result = 0.0
            pred_len = [1]
            actu_len = [1]

            yhat_raw = []
            y = []

            for i in range(num_batch):
                label = labels[i][labels[i]!=-100]
                logit = logits[i][labels[i]!=-100]

                pred = torch.argmax(logit.detach(), axis=1)
                yhat_raw_tmpa, _ = torch.min(logit, axis=0)
                countaa = 1
                for a in pred[1:]:
                    yhat_raw_tmpa[a] = logit[countaa][a]
                    countaa += 1
                yhat_raw.append(yhat_raw_tmpa.detach().tolist())

                y_tmpa = np.zeros(vocab_size)
                for a in label[1:]:
                    y_tmpa[a] = 1
                y.append(y_tmpa)
            yhat_raw = np.array(yhat_raw)
            y = np.array(y)
            ymic = y.ravel()
            
            roc_auc = auc_metrics(yhat_raw, y, ymic)
            pred_len = np.array(pred_len)
            actu_len = np.array(actu_len)
            return torch.tensor(roc_auc["auc_macro"], dtype=torch.float, device=device), torch.tensor(roc_auc["auc_micro"], dtype=torch.float, device=device).sum()

        model = self.model
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)
        else:
            model = self.model
        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.

        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)
        eval_losses: List[float] = []
        preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        accuracy: torch.Tensor = None
        num_accuracy: torch.Tensor = None
        meanap: torch.Tensor = None
        num_meanap: torch.Tensor = None
        meanlen : torch.Tensor = None
        model.eval()

        if is_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        for inputs in tqdm(dataloader, desc=description, mininterval=1200):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                if inputs.get("labels") is not None:
                    labels = inputs["labels"].detach()
                elif inputs.get("masked_lm_labels") is not None:
                    labels = inputs['masked_lm_labels'].detach()
                elif inputs.get("lm_labels") is not None:
                    labels = inputs['lm_labels'].detach()
                else:
                    print('Warning: no label found')

                if is_test:
                    if self.args.load_cui_leng and (not self.args.no_length_planning): 
                        mmap_score_sum, mlen_score_sum = calc_overlap_bart(logits.detach(), labels, device=self.args.device)
                        if meanap is None:
                            meanap = mmap_score_sum
                        else:
                            meanap += mmap_score_sum
                        if num_meanap is None:
                            num_meanap = torch.tensor(1, dtype=torch.long, device=self.args.device)
                        else:
                            num_meanap += torch.tensor(1, dtype=torch.long, device=self.args.device)
                        if meanlen is None:
                            meanlen = mlen_score_sum
                        else:
                            meanlen += mlen_score_sum
                    else:
                        mmap_score_sum, mlen_score_sum = calc_overlap_bart(logits.detach(), labels, device=self.args.device)
                        if meanap is None:
                            meanap = mmap_score_sum
                        else:
                            meanap += mmap_score_sum
                        if num_meanap is None:
                            num_meanap = torch.tensor(1, dtype=torch.long, device=self.args.device)
                        else:
                            num_meanap += torch.tensor(1, dtype=torch.long, device=self.args.device)
                        if meanlen is None:
                            meanlen = mlen_score_sum
                        else:
                            meanlen += mlen_score_sum

                preds = logits[labels!=-100]
                preds = torch.argmax(preds.detach(), axis=1)
                label_ids = labels[labels!=-100]
                if (preds is not None) and (label_ids is not None):
                    if accuracy is None:
                        accuracy = (preds==label_ids).sum()
                    else:
                        accuracy += (preds==label_ids).sum()
                    if num_accuracy is None:
                        num_accuracy = torch.tensor(label_ids.shape[0], dtype=torch.long, device=self.args.device) # number of masked tok
                    else:
                        num_accuracy += torch.tensor(label_ids.shape[0], dtype=torch.long, device=self.args.device)

                if self.args.local_rank < 1 and save_act_predict_result and save_count < 101:
                    save_folder = os.path.join(self.args.output_dir, "eval_results")
                    save_folder = os.path.join(save_folder, "eval_results_{}".format(self.global_step))
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    save_path = os.path.join(save_folder, "{}.txt".format(description))
                    with open(save_path, "a") as writer: 
                        for i in range(labels.shape[0]):
                            writer.write("-----------\n")
                            writer.write("idx {}\n".format(save_count))
                            writer.write(str(inputs["input_ids"][i]))
                            writer.write("\n")
                            writer.write(str(labels[i]))
                            writer.write("\n")
                            if inputs.get("visit_ids") is not None:
                                writer.write(str(inputs["visit_ids"][i]))
                                writer.write("\n")
                            pred = logits[i][labels[i]!=-100]
                            if pred.shape[0] > 0:
                                pred = torch.argmax(pred.detach(), axis=1)
                                writer.write(str(pred))
                                writer.write("\n")
                                label_id = labels[i][labels[i]!=-100]
                                writer.write(str(label_id))
                                writer.write("\n")
                            save_count += 1
        
        if accuracy is not None:
            accuracy = torch.unsqueeze(accuracy,0)
            num_accuracy = torch.unsqueeze(num_accuracy,0)
        if meanap is not None:
            meanap = torch.unsqueeze(meanap,0)
            num_meanap = torch.unsqueeze(num_meanap,0)
        if meanlen is not None:
            meanlen = torch.unsqueeze(meanlen,0)


        if self.args.local_rank != -1:
            # In distributed mode, concatenate all results from all nodes:
            if accuracy is not None:
                accuracy = self.distributed_concat(accuracy, num_total_examples=self.num_examples(dataloader))
            if num_accuracy is not None:
                num_accuracy = self.distributed_concat(num_accuracy, num_total_examples=self.num_examples(dataloader))
            if meanap is not None:
                meanap = self.distributed_concat(meanap, num_total_examples=self.num_examples(dataloader))
            if num_meanap is not None:
                num_meanap = self.distributed_concat(num_meanap, num_total_examples=self.num_examples(dataloader))
            if meanlen is not None:
                meanlen = self.distributed_concat(meanlen, num_total_examples=self.num_examples(dataloader))
        elif is_tpu_available():
            # tpu-comment: Get all predictions and labels from all worker shards of eval dataset
            if accuracy is not None:
                accuracy = xm.mesh_reduce("eval_preds", accuracy, torch.cat)
            if num_accuracy is not None:
                num_accuracy = xm.mesh_reduce("eval_label_ids", num_accuracy, torch.cat)

        # Finally, turn the aggregated tensors into numpy arrays.
        if accuracy is not None:
            accuracy = accuracy.cpu().numpy()
        if num_accuracy is not None:
            num_accuracy = num_accuracy.cpu().numpy()
        if meanap is not None:
            meanap = meanap.cpu().numpy()
        if num_meanap is not None:
            num_meanap = num_meanap.cpu().numpy()
        if meanlen is not None:
            meanlen = meanlen.cpu().numpy()

        metrics = {}
        if accuracy is not None and num_accuracy is not None:
            acc = accuracy.sum() / float(num_accuracy.sum())
            metrics['mask_acc'] = acc
        if is_test:
            if meanap is not None and num_meanap is not None:
                mean_ap = meanap / num_meanap
                metrics['map'] = float(mean_ap[0])
            if meanlen is not None and num_meanap is not None:
                mean_cuilength_in_last_visit = meanlen / num_meanap
                metrics['mean_cuilengthcorrect'] = float(mean_cuilength_in_last_visit[0])  
        
            
        if len(eval_losses) > 0:
            metrics[f"eval_{description}_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_ and description 
        for key in list(metrics.keys()):
            if not key.startswith("eval"):
                metrics[f"eval_{description}_{key}"] = metrics.pop(key)

        return PredictionOutput(predictions=preds, label_ids=label_ids, metrics=metrics)

    def distributed_concat(self, tensor: torch.Tensor, num_total_examples: int) -> torch.Tensor:
        assert self.args.local_rank != -1

        output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output_tensors, tensor)

        concat = torch.cat(output_tensors, dim=0)

        # truncate the dummy elements added by SequentialDistributedSampler
        output = concat[:num_total_examples]
        return output
