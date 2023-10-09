## Exploring Representation-Level Augmentation for Code Search

This repo contains code for [Exploring Representation-Level Augmentation for Code Search](https://arxiv.org/abs/2210.12285), accepted to EMNLP 2022. 
In this codebase we provide instructions for
reproducing our results from the paper. 
We hope that this work can be useful for future 
research on representation-level augmentation.

## Environment

```
conda create -n RACS python=3.6 -y
conda activate RACS
conda install pytorch-gpu=1.7.1 -y
pip install transformers==4.18.0 scikit-learn nltk==3.6.1 tensorboardX tqdm more_itertools pytrec_eval elasticsearch tree_sitter
conda install faiss-cpu -c pytorch
```

## Data
 
We follow the preprocessing rule described in [GraphCodeBERT](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch).
Since our codes are developed based on that repo, you could follow the instructions described in 
[./GraphCodeBERT/README.md](#./GraphCodeBERT/README.md).


## CodeBERT

Here we provide an example of CodeBERT. The augmentaiton methods are described in 
[models.py](#./CodeBERT/models.py). These augmentation methods are orthogonal to any siamese
networks, so you could insert these augmentation code snippets to other models to evaluate the performance.

To train CodeBERT with representation-level augmentation, run:

```angular2html
lang=python
python run_siamese_test.py \
--model_type roberta \
--do_train \
--do_eval \
--do_retrieval \
--evaluate_during_training \
--eval_all_checkpoints \
--data_dir ../data/${lang}/ \
--train_data_file train.jsonl \
--code_type code \
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_retrieval_batch_size 100 \
--learning_rate 1e-6 \
--num_train_epochs 30 \
--gradient_accumulation_steps 1 \
--output_dir ./model/codebert-${lang}-ra \
--encoder_name_or_path microsoft/codebert-base \
--lang ${lang} \
--mix_time 5 \ 
--ra
```

After training, the model will be automatically tested on the test sets.
For detailed configurations, please see the argument parser part in code.

Note: To turn off representation-level augmentation, simply remove ``--ra`` in the command line.

## GraphCodeBERT

Here we provide an example of GraphCodeBERT.
The augmentation codes are the same with CodeBERT.

To train GraphCodeBERT with representation-level augmentation, run:

```angular2html
lang=python
python run.py \
--output_dir=./saved_models/graphbert-${lang}-ra \
--config_name=microsoft/graphcodebert-base \
--model_name_or_path=microsoft/graphcodebert-base \
--tokenizer_name=microsoft/graphcodebert-base \
--lang=${lang} \
--do_train \
--do_eval \
--do_test \
--train_data_file=../data/${lang}/train.jsonl \
--eval_data_file=../data/${lang}/valid.jsonl \
--test_data_file=../data/${lang}/test.jsonl \
--codebase_file=../data/${lang}/codebase.jsonl \
--num_train_epochs 30 \
--code_length 256 \
--data_flow_length 64 \
--nl_length 128 \
--train_batch_size 32 \
--eval_batch_size 64 \
--learning_rate 2e-5 \
--seed 123456 \
--mix_time 5 \
--ra
```

After training, the model will be automatically tested on the test sets.
For detailed configurations, please see the argument parser part in code.

Note: To turn off representation-level augmentation, simply remove ``--ra`` in the command line.


## Citation
If you found this repository useful, please consider citing:
```bash
@inproceedings{li2022exploring,
  title={Exploring Representation-level Augmentation for Code Search},
  author={Li, Haochen and Miao, Chunyan and Leung, Cyril and Huang, Yanxian and Huang, Yuan and Zhang, Hongyu and Wang, Yanlin},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  pages={4924--4936},
  year={2022}
}
```

## Acknowledgement

The implementation of this repo relies on resources from [CoCLR](https://github.com/Jun-jie-Huang/CoCLR), [GraphCodeBERT](https://github.com/microsoft/CodeBERT).
 The code is implemented using PyTorch, We thank the original authors for their open-sourcing.