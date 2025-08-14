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
from utils import load_icr2_gold_retrieval
from utils import retrieval_logits_to_ids

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

    # Decoding Parameters
    parser.add_argument("--max-new-tokens", type=int, default=512, help="HF decoding parameters: deciding how many max new tokens will be produced")
    parser.add_argument("--top-k", type=int, default=4, help="top-k parameter for end2end icr2 tuning.")


    args = parser.parse_args()

    device = "cuda" 
    transformers_token = args.transformers_token
    model_signature = args.model_signature

    # Login hf
    login(transformers_token)

    # Load Models
    if "Mistral" in model_signature:
        model_name = "mistral"
    elif "Phi" in model_signature:
        model_name = "phi"
    elif "Qwen" in model_signature:
        model_name = "qwen"
    else:
        raise NotImplementedError
    
    model = build_model(model_signature, model_name, transformers_token)
    tokenizer = build_tokenizer(model_signature, model_name, transformers_token)


    # Load dataset
    import datasets
    dataset = datasets.load_from_disk("/mnt/task_runtime/data/icr2_train_data_ours/nq-hotpotqa-fever-wow-7500-7500-5000-5000-mistral-retrieve-then-answer-hfcompLoss")
    d_input_ids = dataset[:100]['input_ids']
    d_gold_context_label = dataset[:100]['gold_psg_one_hot']
    
    eval_dataset_list = args.eval_dataset.split(',')
    print(eval_dataset_list)
    for eval_dataset in eval_dataset_list:

        print("*********** Currently Evaluating on "+eval_dataset+" ***********")

        # Load Metrics
        metrics = {}
        
        metrics['em'] = load_metric('em')
        metrics['recallem'] = load_metric('recallem')
        metrics['bidirect_recallem'] = load_metric('bidirect_recallem')
        if "CCI" in model_signature:
            metrics['retrieval_recall_cci'] = load_metric('retrieval_recall_cci')
        elif "RTA" in model_signature:
            metrics['retrieval_recall_rta'] = load_metric('retrieval_recall_rta')
        
        if 'e2e' in model_signature:
            metrics["retrieval_recall_TopK"] = load_metric('retrieval_recall_TopK')

        print("Evaluation metrics:", list(metrics.keys()))

        # Prepare output direcotry
        output_dir = args.experiment_output_path + "/" + eval_dataset + "-" + args.prompt_template
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Load Datasets
        dataset_path = "{benchmark_path}/{eval_dataset}-dev-kilt.jsonl".format(
            benchmark_path = args.benchmark_path, \
            eval_dataset = eval_dataset, \
        )

        # load queries, answers, retrieved corpus
        queries = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                d = json.loads(line.strip())
                queries.append(d)

        if args.validation:
            iter_ = tqdm(queries[:10])
        else:
            iter_ = tqdm(queries)
                
        references = [d['answers'] for d in queries]

        ### Processing Corpus ###
        if args.gold_needle_mode == "random":
            for i, q in enumerate(queries):
                random.shuffle(q['corpus'])
        else:
            raise NotImplementedError
        
        gold_retrieval = [{"gold_context_id": retrieval_logits_to_ids(logit)} for logit in d_gold_context_label]

        print(" *************** Evaluating on {d} ***************".format(d=eval_dataset))

        # Preparing prompt template
        sys_prompt_template = load_template(model_name+"_sys")
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
        if "e2e" in model_signature:
            # Append [CONTEXT] and concatenate [/CONTEXT] to each context item
            prompt_template.set_context_special_token()
        
        model_predictions = []
        retrieval_predictions, answer_predictions, retrieval_topk_ids = [], [], []
        eval_results = {}

        if "e2e" in model_signature:
            ret_model, gen_model = model
        
        with torch.no_grad():
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

                # Adding the system prompt
                prompt = sys_prompt_template(prompt)

                if i == 0:
                    with open(output_dir + "/prompt_example.txt", "w+", encoding='utf-8') as f:
                        f.writelines([prompt])

                # model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
                # d_input_ids[i] = d_input_ids[i][:d_input_ids[i].index(4) + 1]
                # print(d_input_ids[i])
                
                # print("total length:", len(d_input_ids[i]), "; Last 20 positions:", d_input_ids[i][-20:])
                model_inputs = torch.tensor(d_input_ids[i]).unsqueeze(0).to(device)

                if "e2e" in model_signature and "token" in model_signature:
                    
                    retrieval_mask, retrieval_logits = ret_model.forward(
                        input_ids=model_inputs, 
                        gumbel_tau=1.0,
                        gumbel_k=args.top_k,
                        )
                    
                    retrieval_ids = torch.masked_select(model_inputs, retrieval_mask.bool()).unsqueeze(0).to(torch.long) # num_context, max_context_len, d -> top_k=4, max_context_len, d
                    retrieval_psg_ids = retrieval_logits_to_ids(retrieval_logits[0])
                    gold_psg_ids = retrieval_logits_to_ids(d_gold_context_label[i])

                    generated_ids = gen_model.generate(
                        input_ids=retrieval_ids, 
                        max_new_tokens=args.max_new_tokens,
                    )

                    print("retrieval IDS:", retrieval_psg_ids)
                    print("gold IDS:", gold_psg_ids)

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(retrieval_ids, generated_ids)
                    ]
                else:
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=args.max_new_tokens,
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                model_predictions.append(response) # Save model's actual predictions to predictions files
                retrieval_topk_ids.append(retrieval_psg_ids)

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

            with open(output_dir + "/predictions.txt", "w+", encoding='utf-8') as f:
                f.writelines([p.replace('\n', '').strip()+'\n' for p in model_predictions])

            with open(output_dir + '/eval_result.json', 'w') as f:
                json.dump(eval_results, f, indent=4)

            

        
        
