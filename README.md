# ICR2: Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models
This software accompanies the paper [Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models
](https://arxiv.org/abs/2501.08248)

Yifu Qiu, Varun Embar, Yizhe Zhang, Navdeep Jaitly, Shay B. Cohen, Benjamin Han

ACL Findings, 2025

Please cite our paper if you find our work useful for your research:
```
@article{qiuacl25,
  title={Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models}, 
  author={Yifu Qiu and Varun Embar and Yizhe Zhang and Navdeep Jaitly and Shay B. Cohen and Benjamin Han},
  year={2025},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2025},
}
```

## Installation

## Create environment

The environments file `icr_env.yaml` and `easycontext_env.yaml` contain the full environment configuration, including all packages, dependencies, and the exact versions.
Create the two environments by,
```
conda env create -f icr_env.yaml
conda env create -f easycontext_env.yaml
```

## Install additional packages

Some additional packages (e.g., DeepSpeed) need to be configured. Simply run,
`bash setup.bash`

which will install the following packages,

1. DeepSpeed (from local, because its latest version has conflict with easycontext implementation)
2. seaborn (for visualizing attention head analysis)
3. prettytable (for visualizing attention head analysis)

## Creation of ICR2 dataset
To create the ICR2 benchmark, we will need the following data:

0. Download the kilt dataset using the steps described here: `https://ai.meta.com/tools/kilt/`
1. Generate gold passage parsed from KILT's human annotation using: `./datagen/extract_gold_psg.py`
2. Generate negative passages retrieved from 5 retrievers. `./datagen/make_neg_psg.py`

Then you can simply create the dataset by,

```
python scripts/icr2_data_creation.py \
    --context-length-num-passage 200 \
    --datasets nq,hotpotqa,fever,wow \
    --models dpr_our,contriever_our,bm25_our,drqa,blink 
```

`--context-length-num-passage 200` controls the length of overall context. 200 is set if the target context is 32K, 100 is for 16K, 500 for 64K, 700 for 1M.

`--models dpr_our,contriever_our,bm25_our,drqa,blink` indicates what retrievers we want to use.

`--datasets` indicates what subsets of kilt are used.

## Pre-tokenization for training set

To improve the training efficiency, we pre-tokenize all instances in training set into tokens. This can be done by,

```
MODEL_SIGNATURE="mistralai/Mistral-7B-Instruct-v0.3"

python ./src/train/dataset_processing/icr2_dataset.py \
    --datasets nq,hotpotqa,fever,wow \
    --data-size 7500,7500,5000,5000 \
    --model-signature $MODEL_SIGNATURE \ # using model-specific tokenizer
    --dataset-read-path "data/icr2_benchmark/" \ # raw datafiles path
    --dataset-save-path "data/icr2_train_data/" \ # path for saving tokenized training data
    --max-seq-length 32768 \ # max_sequence_length for models, e.g., mistral only accepts 32k tokens
    --sft-mode "retrieve-then-answer" \ # mode for sft, can be 'direct-answer' or 'cite-context-id'
    --dataset-type "sft-v2" # sft-v2 for adding [RETRIEVAL][/RETRIEVAL] to model's output phase, can be 'sft' or 'icr2-tuning'
```

This step might takes 1 hour to prepare a training set with 25K samples.

## Run training

### Running command

The following command is an example for SFT training a retrieve-then-answer model, which is expected to run with 8xA100-40G GPUs requiring 1024G storage for saving 6 intermediate checkpoints safely.

```
TRAIN_PATH=./src/train/
HOME_PATH=./

MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"

EXPERIMENT_NAME="Mistral-0.3-7B-Instrcuct-rope-1m-bs32-cosineLR1e5-warmup0.1-update2400-32k-n7500-h7500-f5000-w5000-RTA"
accelerate launch \
--config_file  ${TRAIN_PATH}/easy_context/accelerate_configs/single_node.yaml \
${TRAIN_PATH}/train.py \
--batch-size 1 \
--gradient-accumulate-every 4 \
--seed 2024 \
--wandb ICR2-sft-v2 \
--output-dir ${HOME_PATH}/exp_results/training_output/${EXPERIMENT_NAME} \
--max-train-steps 2400  \
--learning-rate 1e-5  \
--dataset data/icr2_train_data/nq-hotpotqa-fever-wow-7500-7500-5000-5000-mistral-retrieve-then-answer-hf-v2/ \
--model ${MODEL_NAME}  \
--seq-length 32768 \
--rope-theta 1000000 \
--parallel_mode data_parallel \
--checkpointing-steps 400 \
```

### Details about calculations for batch size and epochs

If we have 8 GPUs and each of them contains one instance and set `gradient-accumulate-every=4`, so the actual batch size is `32`. 

We have 25K training instances. If we want to update for approximate `3` epochs, the `max-train-steps` will be `25000 / 32 * 3 ~= 2400`.

**Warning: If you want to change `max-train-steps`, remember to also change it in `src/train/easy_context/accelerate_configs/zero3_offload.json` for DeepSpeed's configuration**

## Run Inference

An example evaluation run for the ICR2 benchmark is given below.

### Evaluation on ICR2
```
python src/evaluations/icr2_eval.py \
        --model-signature ${MODEL_SIGNATURE} \
        --transformers-token ${TRANSFORMERS_TOKEN} \
        --experiment-output-path ${OUTPUT_PATH} \
        --eval-dataset nq,hotpotqa,fever,wow \
        --prompt-template rag \
        --gold-needle-mode random \
        --max-new-tokens 1024 \
```

`--model-signature` accepts the huggingface's model's signature. For example, when we use `mistralai/Mistral-7B-Instruct-v0.3`, we will download the corresponding model from huggingface and execute the evaluation.

We can also set the variable of `--output-dir` in the previous training script to execute the evaluation with our trained model, for example, `"exp_results/training_output/Mistral-0.3-7B-Instrcuct-rope-1m-bs32-cosineLR1e5-warmup0.1-update2400-32k-n7500-h7500-f5000-w5000-RTA/step_1600"`.

`--transformers-token` is your huggingface user account's token which is necessary for some "protected" open-sourced model such as Mistral. More details are in [here](https://huggingface.co/docs/hub/en/security-tokens). 

`--max-new-tokens` is set to `1024`. This is especially for RTA model, if `max_new_tokens` is too small, model's generation may be cut off, and model never tries to generate answer after retrieval phase.

`--gold-needle-mode` is set to `random`. This controls where to place the gold passages. We randomly insert the gold passages into other negative passages.

`--prompt-template` controls how we evaluate the model. It can be `oracle`, `closebook` or `rag`. 


### Probing Attention Heads for Retrieval


`--probe-attention` used only for DA model. If it is true, model will first probe attention heads on validation set, then using the top-M context (`--head-topk`) in top-K (`--probe-attn-k-head`) attention heads for explicitly retrieving contexts. An example for probing 4 attention heads with each heads looking at 8 context will be,

```
python src/evaluations/loft_eval.py \
        --model-signature ${MODEL_SIGNATURE} \
        --transformers-token ${TRANSFORMERS_TOKEN} \
        --experiment-output-path ${OUTPUT_PATH} \
        --eval-dataset nq,hotpotqa,musique \
        --eval-context-length 32k-ours \
        --prompt-template rag-qa \
        --max-new-tokens 1024 \
        --probe-attention \
        --probe-attn-k-head 4 \
        --head-topk 8 \
```

I suggest to only use probing attention head retrieval on DA model. Though I have implemented it for RTA and CCI, it would take much more computation on RTA & CCI: in DA we track the attention scores for ranking only on the first token, while RTA and CCI need to average the attention scores during tracking all tokens inside `<RETRIEVAL>` and `</RETRIEVAL>`.  

### ICR2 Model Inference

For ICR2 model (using an explicit Gumbel-TopK retrieval head), we need to specify the following argument in evaluation scripts. This controls how many contexts we want the topK operator to select in testing time.

```
parser.add_argument("--top-k", type=int, default=4, help="top-k parameter for end2end icr2 tuning.")
```

For example, if `--top-k 4` is set, 4 passages will be selected by retrieval head, and they will be presented for model's generation.

### Evaluation on LOFT

In LOFT, we simply change the evaluation scripts from `icr2_eval.py` to `loft_eval.py`.

```
python src/evaluations/loft_eval.py \
        --model-signature ${MODEL_SIGNATURE} \
        --transformers-token ${TRANSFORMERS_TOKEN} \
        --experiment-output-path ${OUTPUT_PATH} \
        --eval-dataset nq,hotpotqa,musique \
        --eval-context-length 32k-ours \
        --prompt-template rag-qa \
        --max-new-tokens 1024 \
```

`--prompt-template` Note that compared to icr2 evaluation, loft should contain `-qa` as a postfix there, because loft can be used for various tasks, and we only use qa here.

I put all evaluation scripts for reproducing results on `scripts/inference.bash`.

### Other args

As we have few new methods on inference, so there are other args should be considered.

`--block-rta-context-decoding` used only for RTA model. If it is true, when model decodes tokens after `[RETRIEVAL][/RETRIEVAL]`, it can only attend to tokens inside `[RETRIEVAL][/RETRIEVAL]` rather than the whole context.


### Path for saved evaluation scripts

All response predictions, evaluation results will be saved inside `exp_results/{timestamp_for_evaluation}/{model_signature}/{dataset}` such as `exp_results/20240627_02:09:12-loft/Qwen/Qwen2-7B-Instruct/nq`.
