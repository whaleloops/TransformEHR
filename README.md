# TransformEHR

This repository provides the code for fine-tuning TransformEHR, a generative encoder-decoder model with transformer that was pretrained using a new pretraining objective - predicting all diseases and outcomes of a patient at a future visit from previous visits. 


## Dependencies

* Operating systems: Ubuntu 20.04.5 LTS
* Python 3.8.11 with libraries:
* [NumPy](http://www.numpy.org/) (currently tested on version 1.20.3)
* [PyTorch](http://pytorch.org/) (currently tested on version 1.9.0+cu111)
* [Transformers](https://github.com/huggingface/transformers) (currently tested on version 4.16.2)
* tqdm==4.62.2
* scikit-learn==0.24.2

## Installation
 
Use pip to install, typical install time is about 30 minutes.

## How to load sample data

An example to load sample data is located at sample_load.py

## How to fine-tune

CUDA_VISIBLE_DEVICES=0 python main_bart.py \
                --do_pos_emb --icd_training --load_cui_leng \
                --output_dir PATHTOOUTPUT \
                --logging_dir PATHTOLOG  \
                --model_type=bart --config_name=PATHTOLOAD \
                --do_train --train_data_file=PATHTOTRAIN \
                --per_device_train_batch_size 24 \
                --do_eval --eval_data_file=PATHTODEV \
                --per_device_eval_batch_size 48 \
                --test_data_file=PATHTOTEST \
                --block_size 512 \
                --mlm --mlm_probability 0.5 \
                --logging_first_step \
                --logging_steps 470 --save_steps 100000 \
                --num_train_epochs 12 \
                --learning_rate 1e-4 \
                --warmup_steps 470 \
                --weight_decay 0.05

The code will print the following to the standard output of the system: the AUROC, AUPRC, and the correspoding step for dev and eval data.

The time to run finetuning depends on the size of the data. Please checkout the Implementation Details section in the paper for more detail.
