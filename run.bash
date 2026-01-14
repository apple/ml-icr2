#!/bin/bash

. /etc/profile.d/conda.sh
zconda activate icr

export PYTHONPATH="/mnt/task_runtime/"
export WANDB_DISABLE=true
export HF_HOME="./transformers-home"
export PYTHONIOENCODING=UTF-8

TRANSFORMERS_TOKEN=<YOUR HF TOKEN HERE>

MODEL_SIGNATURE="Qwen/Qwen2-7B-Instruct"
# MODEL_SIGNATURE="Qwen/Qwen2-1.5B-Instruct"
# MODEL_SIGNATURE="mistralai/Mistral-7B-Instruct-v0.3"
# MODEL_SIGNATURE="microsoft/Phi-3-small-128k-instruct"

### Setup for running experiment
TS=$(date "+%Y%0m%0d_%T")
OUTPUT_PATH="./exp_results/${TS}-loft/"${MODEL_SIGNATURE}"/"
mkdir -p $OUTPUT_PATH

python src/evaluations/loft_eval.py \
    --model-signature ${MODEL_SIGNATURE} \
    --transformers-token ${TRANSFORMERS_TOKEN} \
    --experiment-output-path ${OUTPUT_PATH} \
    --eval-dataset nq,hotpotqa,musique \
    --eval-context-length 32k-ours \
