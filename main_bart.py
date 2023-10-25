# coding=utf-8
# Modified from transformer library by The Google AI Language Team Authors and The HuggingFace Inc. team.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    TextDataset,
    TrainingArguments,
    set_seed,
)
from trainer import Trainer

from transformers.trainer_utils import EvalPrediction, PredictionOutput, TrainOutput

from dataset import DataCollatorForICDBERT, DataCollatorForICDBERTFINALPRED, DataCollatorForICDBART, prepare_dataset
from icdmodelbart import ICDBartForPreTraining


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    icd_training: bool = field(
        default=False,
        metadata={"help": "Whether train as a icd training"},
    )
    do_multi: bool = field(
        default=False,
        metadata={"help": "Whether train as a icd training"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with bart masking by natural visit, instead of to mask with poisoon_random lambda 4"}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    do_pos_emb: bool = field(
        default=False,
        metadata={"help": "Whether to do positional embedding "},
    )
    debug_mode: bool = field(
        default=False,
        metadata={"help": "Debug or not"},
    )
    save_mode: bool = field(
        default=False,
        metadata={"help": "Create data from scracth"},
    )
    load_cui_leng: bool = field(
        default=False,
        metadata={"help": "Weather to use or load 00_cuilengs.pt for span/bart training"},
    )
    no_length_planning: bool = field(
        default=False,
        metadata={"help": "Introduces a new latent variable z that controls the length of span, false is to introduce"},
    )
    do_poisoon_random_masking: bool = field(
        default=False,
        metadata={"help": "Wheather to mask with poisoon_random lambda 4 instead by natural visit"},
    )
    do_date_visit: bool = field(
        default=False,
        metadata={"help": "Wheather to treat each visit embedding as a date(new) or a seqnence number(like bert)"},
    )


# def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, evaluate=False):
#     file_path = args.eval_data_file if evaluate else args.train_data_file
#     if args.line_by_line:
#         return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=args.block_size)
#     else:
#         return TextDataset(
#             tokenizer=tokenizer, file_path=file_path, block_size=args.block_size, overwrite_cache=args.overwrite_cache
#         )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.tokenizer_name:
        tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )    

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
    config.date_visit_embeddings = data_args.do_date_visit


    if model_args.icd_training:
        if model_args.model_name_or_path:
            model = ICDBartForPreTraining.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Training new ICD model from scratch")
            model = ICDBartForPreTraining(config)
    else:
        if model_args.model_name_or_path:
            model = AutoModelWithLMHead.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelWithLMHead.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = data_args.block_size

    # Get datasets
    train_dataset, eval_dataset, test_datasets = prepare_dataset(model_args, data_args, training_args, tokenizer=tokenizer)

    # train_dataset = get_dataset(data_args, tokenizer=tokenizer) if training_args.do_train else None
    # eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    if data_args.load_cui_leng:
        data_collator = DataCollatorForICDBART(
            tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability, no_length_planning=data_args.no_length_planning,
        )
        test_data_collator = DataCollatorForICDBART(
            tokenizer=tokenizer, mlm=True, mlm_probability=-0.1, decode_targets_masked_only=True, no_length_planning=data_args.no_length_planning
        )
    else:

        if model_args.do_multi:
            data_collator = DataCollatorForICDBERTFINALPRED(
                tokenizer=tokenizer, numlastvisit=1, do_pos_emb=data_args.do_pos_emb, for_bert=False
            )
        else:
            data_collator = DataCollatorForICDBERT(
                tokenizer=tokenizer, mlm=data_args.mlm, mlm_probability=data_args.mlm_probability, do_pos_emb=data_args.do_pos_emb, for_bert=False
            ) 
        test_data_collator = DataCollatorForICDBERTFINALPRED(
            tokenizer=tokenizer, numlastvisit=1, do_pos_emb=data_args.do_pos_emb, for_bert=False
        )
    

    # print("len(train_dataset)")
    # print(len(train_dataset))
    # print("len(eval_dataset)")
    # print(len(eval_dataset))

    # def compute_metrics(p:EvalPrediction)-> Dict:
    #     return {"acc": simple_accuracy(p.predictions, p.label_ids)}

    # Initialize our Trainer
    training_args.load_cui_leng = data_args.load_cui_leng
    training_args.no_length_planning = data_args.no_length_planning
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        test_collator=test_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_datasets=test_datasets,
        prediction_loss_only=False
    )


    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        perplexity = math.exp(eval_output["eval_00_valid_loss"])
        eval_output["perplexity"] = perplexity

        output_eval_file = os.path.join(training_args.output_dir, "eval_results_lm.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key in sorted(eval_output.keys()):
                    logger.info("  %s = %s", key, str(eval_output[key]))
                    writer.write("%s = %s\n" % (key, str(eval_output[key])))

        results.update(eval_output)

    # if training_args.do_predict:
    #     for test_dataset in test_datasets:
    #         prediction_output = trainer.predict(test_dataset=test_dataset)
    #         predictions = np.argmax(prediction_output.predictions, axis=1) 
    #         label_ids = prediction_output.label_ids
    #     if trainer.is_world_master():


    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
