#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, './src') # add src path
import os

from transformers import set_seed
from models import build_model, build_tokenizer
from evaluator.utils import load_metric
from prompt.utils import load_template

import torch
import json
import argparse
from tqdm import tqdm
import random
from huggingface_hub import login

from utils import retrieval_response_post_processing
from utils import load_icr2_dataset, load_icr2_metrics



import openai 

openai.api_key = "<OPENAIKEY>"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    set_seed(42)  # For reproducibility in decoding

    parser.add_argument("--model-signature", type=str)
    parser.add_argument("--transformers-token", type=str)
    parser.add_argument("--benchmark-path", type=str, default='./data/icr2_benchmark/')
    parser.add_argument("--eval-dataset", type=str, default='nq')
    parser.add_argument("--experiment-output-path", type=str)

    # Evaluation Parameters
    parser.add_argument("--prompt-template", type=str, default='rag', choices=['rag', 'closebook', 'oracle'])
    parser.add_argument("--gold-needle-mode", type=str, default='random', help="Deciding where to place the gold passages")

    # Experiment Parameters
    parser.add_argument("--validation", action="store_true", help="set True to use validation set (for dubugging/development purpose)") 

    # Flag for tweaking RTA decoding, in answering phase, we only allow model to attend decoded tokens in retrieval phase, rather than context
    parser.add_argument("--block-rta-context-decoding", action="store_true", help="we only allow model to attend decoded tokens in retrieval phase, rather than context") 
    parser.add_argument("--hierarchical-mask-decoding", action="store_true", help="Use for HierAttnMask model.") 
    

    parser.add_argument("--probe-attention", action="store_true", help="set True to probe attention head for retrieval)") 
    parser.add_argument("--probe-attn-k-head", type=int, default=4, help="How many attention heads are used to retrieve")
    parser.add_argument("--probe-attn-k-threshold", type=float, default=0.0, help="Only use attention head with score higher than threshold")
    parser.add_argument("--head-topk", type=int, default=4, help="top-k parameter for each attention head to consider.")

    # Decoding Parameters
    parser.add_argument("--max-new-tokens", type=int, default=512, help="HF decoding parameters: deciding how many max new tokens will be produced")
    parser.add_argument("--top-k", type=int, default=4, help="top-k parameter for end2end icr2 tuning.")


    args = parser.parse_args()

    device = "cuda" 
    transformers_token = args.transformers_token
    model_signature = args.model_signature
    
    model = None
    tokenizer = None

    # for v2 sft
    if "CCI" in model_signature or "RTA" in model_signature:
        RETRIEVAL_TOKEN_START = "[RETRIEVAL]"
        RETRIEVAL_TOKEN_END = "[/RETRIEVAL]"

        special_tokens_dict = {'additional_special_tokens': [RETRIEVAL_TOKEN_START, RETRIEVAL_TOKEN_END]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"{num_added_toks} new tokens have been added...")
        # model.resize_token_embeddings(len(tokenizer))

    eval_dataset_list = args.eval_dataset.split(',')
    print(eval_dataset_list)
    for eval_dataset in eval_dataset_list:

        print("*********** Currently Evaluating on "+eval_dataset+" ***********")

        metrics = load_icr2_metrics(args, eval_dataset=eval_dataset, model_signature=model_signature)

        print("Evaluation metrics:", list(metrics.keys()))

        # Prepare output direcotry
        output_dir = args.experiment_output_path + "/" + eval_dataset + "-" + args.prompt_template
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Save the dictionary to a JSON file
        with open(output_dir+"/eval_config.json", 'w') as json_file:
            json.dump(vars(args), json_file, indent=4)

        # Load Datasets
        iter_, references, gold_retrieval, gold_context_index = load_icr2_dataset(args, eval_dataset=eval_dataset, split='dev')

        # Probe attention head here
        # On validation set we find which attention heads have highest recall rate
        if args.probe_attention:
            valid_iter_, _, _, valid_gold_context_index = load_icr2_dataset(args, eval_dataset=eval_dataset, split='valid')

        print(" *************** Evaluating on {d} ***************".format(d=eval_dataset))

        # Preparing prompt template
        if eval_dataset == "nq" or eval_dataset == "hotpotqa" or eval_dataset == "triviaqa":
            prompt_template = load_template(args.prompt_template + '-qa')
        elif eval_dataset == "fever":
            prompt_template = load_template(args.prompt_template + '-fact-check')
        elif eval_dataset == "wow":
            prompt_template = load_template(args.prompt_template + '-dialogue')
        else:
            raise NotImplementedError
        
        if "CCI" in model_signature:
            # For CCI model, we need to add citation mark in prompt
            prompt_template.set_document_id()
        # if "e2e" in model_signature:
        #     # Append [CONTEXT] and concatenate [/CONTEXT] to each context item
        #     prompt_template.set_context_special_token()
        
        model_predictions = []
        retrieval_predictions, answer_predictions, retrieval_topk_ids, attention_retrieval = [], [], [], []
        eval_results = {}

        for i, d in enumerate(iter_):
            
            torch.cuda.empty_cache()

            # Adding the context into prompt
            if "rag" in args.prompt_template:
                prompt = prompt_template(d, d['corpus'])
            elif "closebook" in args.prompt_template:
                prompt = prompt_template(d)
            elif "oracle" in args.prompt_template:
                prompt_template.extract_oracle_for_icr2(d['corpus'])
                prompt = prompt_template(d)
            else:
                raise NotImplementedError

            if i == 0:
                with open(output_dir + "/prompt_example.txt", "w+", encoding='utf-8') as f:
                    f.writelines([prompt])

            try:
                # Call the OpenAI API
                response = openai.ChatCompletion.create(
                    model=args.model_signature,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=args.max_new_tokens,
                    temperature=0.7
                )

                # Extract and return the model's answer
                response = response['choices'][0]['message']['content']

            except Exception as e:
                import time
                time.sleep(30)

                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=args.max_new_tokens,
                    temperature=0.7
                )

                # Extract and return the model's answer
                response = response['choices'][0]['message']['content']


            model_predictions.append(response) # Save model's actual predictions to predictions files


            if "CCI" in model_signature or "RTA" in model_signature:
                # For chain-of-thought response, process to get retrieval items and predicted answer 
                retrieval_prediction, predicted_answer = retrieval_response_post_processing(args, eval_dataset, response)
            else:
                retrieval_prediction, predicted_answer = None, response
            
            answer_predictions.append(predicted_answer) # Save the post-processed predictions for evaluation
            retrieval_predictions.append(retrieval_prediction) # Save the post-processed predictions for evaluation

        for m in metrics:
            if "retrieval" not in m:
                eval_results[m] = metrics[m](predictions=answer_predictions, references=references)
            else:
                if "CCI" in model_signature:
                    eval_results["retrieval_recall_cci"] = metrics["retrieval_recall_cci"](retrieval_prediction=retrieval_predictions, gold_retrieval=gold_retrieval)
                elif "RTA" in model_signature:
                    eval_results["retrieval_recall_rta"] = metrics["retrieval_recall_rta"](retrieval_prediction=retrieval_predictions, gold_retrieval=gold_retrieval)
                if 'e2e' in model_signature:
                    eval_results["retrieval_recall_TopK"] = metrics["retrieval_recall_TopK"](retrieval_psg_ids=retrieval_topk_ids, gold_retrieval=gold_retrieval)
                if args.probe_attention:
                    eval_results["retrieval_probe_attention"] = metrics["retrieval_probe_attention"](retrieval_prediction=attention_retrieval, gold_retrieval=gold_retrieval)

        with open(output_dir + "/predictions.txt", "w+", encoding='utf-8') as f:
            f.writelines([p.replace('\n', '').strip()+'\n' for p in model_predictions])

        with open(output_dir + '/eval_result.json', 'w') as f:
            json.dump(eval_results, f, indent=4)

        

    
    
