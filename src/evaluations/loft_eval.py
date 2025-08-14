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
from huggingface_hub import login

from utils import retrieval_response_post_processing
from utils import load_loft_gold_retrieval
from utils import retrieval_logits_to_ids
from train.utils import block_ctx_for_query_attention_mask
from utils import load_icr2_dataset, parse_special_tokens_for_probe_attn, prob_attn_calculate, calculate_element_distance, process_retrieval_score, prob_attn_retrieve, decode_with_prefix_states

set_seed(42)  # For reproducibility in decoding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-signature", type=str)
    parser.add_argument("--transformers-token", type=str)
    parser.add_argument("--benchmark-path", type=str, default='./data/loft/')
    parser.add_argument("--eval-task", type=str, default='rag')
    parser.add_argument("--eval-dataset", type=str, default='nq')
    parser.add_argument("--eval-context-length", type=str, default='32k')
    parser.add_argument("--experiment-output-path", type=str)

    # Evaluation Parameters
    parser.add_argument("--prompt-template", type=str, default='rag-qa', choices=['rag-qa', 'closebook-qa', 'oracle-qa'])
    parser.add_argument("--gold-needle-mode", type=str, default='random', help="Deciding where to place the gold passages")

    # Experiment Parameters
    parser.add_argument("--validation", action="store_true", help="set True to use validation set (for dubugging/development purpose)") 
    parser.add_argument("--probe-attention", action="store_true", help="set True to probe attention head for retrieval)") 
    parser.add_argument("--probe-attn-k-head", type=int, default=4, help="How many attention heads are used to retrieve")
    parser.add_argument("--probe-attn-k-threshold", type=float, default=0.0, help="Only use attention head with score higher than threshold")
    parser.add_argument("--head-topk", type=int, default=4, help="top-k parameter for each attention head to consider.")

    parser.add_argument("--block-rta-context-decoding", action="store_true", help="we only allow model to attend decoded tokens in retrieval phase, rather than context") 
    parser.add_argument("--hierarchical-mask-decoding", action="store_true", help="Use for HierAttnMask model.") 
    
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
    
    model = build_model(model_signature, model_name, transformers_token, args)
    tokenizer = build_tokenizer(model_signature, model_name, transformers_token)

    # for v2 sft
    if "CCI" in model_signature or "RTA" in model_signature:
        RETRIEVAL_TOKEN_START = "[RETRIEVAL]"
        RETRIEVAL_TOKEN_END = "[/RETRIEVAL]"

        special_tokens_dict = {'additional_special_tokens': [RETRIEVAL_TOKEN_START, RETRIEVAL_TOKEN_END]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        print(f"{num_added_toks} new tokens have been added...")
        # model.resize_token_embeddings(len(tokenizer))

    
    eval_dataset_list = args.eval_dataset.split(',')
    for eval_dataset in eval_dataset_list:

        # Load Metrics
        metrics = {}
        if eval_dataset == "nq" or eval_dataset == "hotpotqa" or eval_dataset == "musique":
            metrics['em'] = load_metric('em')
            metrics['recallem'] = load_metric('recallem')
            metrics['bidirect_recallem'] = load_metric('bidirect_recallem')
            if "CCI" in model_signature:
                metrics['retrieval_recall_cci'] = load_metric('retrieval_recall_cci')
            elif "RTA" in model_signature:
                metrics['retrieval_recall_rta'] = load_metric('retrieval_recall_rta')
        else:
            print("Dataset is not in nq,hotpotqa,musique")
            raise NotImplementedError
        if 'e2e' in model_signature:
            metrics["retrieval_recall_TopK"] = load_metric('retrieval_recall_TopK')
        if args.probe_attention:
            metrics["retrieval_probe_attention"] = load_metric('retrieval_probe_attention')

        # Prepare output direcotry
        output_dir = args.experiment_output_path + "/" + eval_dataset + "-" + args.prompt_template
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        # Save the dictionary to a JSON file
        with open(output_dir+"/eval_config.json", 'w') as json_file:
            json.dump(vars(args), json_file, indent=4)

        # Load Datasets
        dataset_path = "{benchmark_path}/{eval_task}/{eval_dataset}/{eval_context_length}".format(
            benchmark_path = args.benchmark_path, \
            eval_task = args.eval_task, \
            eval_dataset=eval_dataset, \
            eval_context_length=args.eval_context_length
        )

        # load retrieved corpus
        retrieved_corpus_path = dataset_path + '/corpus.jsonl'
        with open(retrieved_corpus_path, 'r', encoding='utf-8') as f:
            corpus = [json.loads(l) for l in f]

        # load dev/testing queries for debugging
        if args.validation:
            queries_path = dataset_path + '/dev_queries.jsonl'
        else:
            queries_path = dataset_path + '/test_queries.jsonl'
        
        # Load reference and gold context ID and text
        references = []
        with open(queries_path, 'r', encoding='utf-8') as f:
            queries = [json.loads(l) for l in f]
            references = [d['answers'] for d in queries]

        gold_retrieval = load_loft_gold_retrieval(corpus, queries)

        # load fewshot queries
        few_shot_queries = dataset_path + '/few_shot_queries.jsonl'
        with open(few_shot_queries, 'r', encoding='utf-8') as f:
            few_shot_queries = [json.loads(l) for l in f]

        print(" *************** Evaluating on {d} ***************".format(d=eval_dataset))
        print(corpus[0])
        print(queries[0])
        print(few_shot_queries[0])

        # Preparing prompt template
        sys_prompt_template = load_template(model_name+"_sys")
        if args.prompt_template == "rag-qa":
            prompt_template = load_template("rag-qa")
        elif args.prompt_template == "closebook-qa":
            prompt_template = load_template("closebook-qa")
        elif args.prompt_template == "oracle-qa":
            prompt_template = load_template("oracle-qa")
            prompt_template.build_oracle_corpus(dataset_path + "/corpus.jsonl")
        else:
            raise NotImplementedError
        
        if "CCI" in model_signature:
            # For CCI model, we need to add citation mark in prompt
            prompt_template.set_document_id()
        # if "e2e" in model_signature:
        #     # Append [CONTEXT] and concatenate [/CONTEXT] to each context item
        #     prompt_template.set_context_special_token()

        # Probe attention head here
        # On validation set we find which attention heads have highest recall rate
        if args.probe_attention:
            if eval_dataset == "musique":
                eval_dataset = "hotpotqa" # using hotpotqa's validation set to find heads for musique
            valid_iter_, _, _, valid_gold_context_index = load_icr2_dataset(args, eval_dataset=eval_dataset, split='valid')

        model_predictions = []
        retrieval_predictions, answer_predictions, retrieval_topk_ids, attention_retrieval = [], [], [], []
        eval_results = {}

        retrieval_score = [[[0, ''] for _ in range(model.config.num_hidden_layers)] for _ in range(model.config.num_attention_heads)]
        with torch.no_grad():

            if args.probe_attention:
                print("Probing the attention head for maximizing retrieval recall on validation set...")
                avg_response_len = 0
                attn_head_hit = 0
                for val_i, val_d in enumerate(valid_iter_):
                    torch.cuda.empty_cache()    

                    # Adding the context into prompt
                    if "rag" in args.prompt_template:
                        prompt = prompt_template(val_d, val_d['corpus'])
                    elif "closebook" in args.prompt_template:
                        prompt = prompt_template(d)
                    elif "oracle" in args.prompt_template:
                        prompt_template.extract_oracle_for_icr2(d['corpus'])
                        prompt = prompt_template(d)
                    else:
                        raise NotImplementedError

                    # Adding the system prompt
                    prompt = sys_prompt_template(prompt)
                    valid_model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

                    input_ids = valid_model_inputs.input_ids
                    input_ids_ = input_ids.tolist()[0]
                    prompt_enc_states = model(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)

                    try:
                        special_tok_positions = parse_special_tokens_for_probe_attn(model_inputs=valid_model_inputs, model_signature=model_signature, eval_dataset=eval_dataset, corpus=val_d['corpus'])
                    except:
                        print("Skip eval for parsing error...")
                        continue

                    needle_start, needle_end = [], []
                    for g_ctx_ids in valid_gold_context_index[val_i]:
                        needle_start.append(special_tok_positions[g_ctx_ids])
                        needle_end.append(special_tok_positions[g_ctx_ids + 1])

                    generated_ids, retrieval_score = prob_attn_calculate(model, tokenizer, model.config, prompt_enc_states, input_ids[:,-1], args.max_new_tokens, needle_start, needle_end, retrieval_score)

                    if "CCI" in model_signature or "RTA" in model_signature:
                        avg_response_len += calculate_element_distance(generated_ids, 32768, 32769)
                    else:
                        avg_response_len += len(generated_ids)

                sorted_ret_head = process_retrieval_score(args, retrieval_score, avg_response_len, model.config, eval_dataset, output_dir, top_k_head=args.probe_attn_k_head) # normalize, visualize, then save retrieval_score and output ranked topk attention head index
            
            for i, d in enumerate(tqdm(queries)):
                
                torch.cuda.empty_cache()

                # Adding the context into prompt
                if args.prompt_template == "rag-qa":
                    prompt = prompt_template(d, corpus)
                elif args.prompt_template == "closebook-qa":
                    prompt = prompt_template(d)
                elif args.prompt_template == "oracle-qa":
                    prompt = prompt_template(d)
                else:
                    raise NotImplementedError

                # Adding the system prompt
                prompt = sys_prompt_template(prompt)

                if i == 0:
                    with open(output_dir + "/prompt_example.txt", "w+", encoding='utf-8') as f:
                        f.writelines([prompt])

                model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

                if "e2e" in model_signature and "token" not in model_signature:
                    ret_model, gen_model = model

                    retrieval_ids, retrieval_logits = ret_model.forward(
                        input_ids=model_inputs.input_ids, 
                        attention_mask=model_inputs.attention_mask,
                        gumbel_tau=1.0,
                        gumbel_k=args.top_k,
                    )
                    
                    retrieval_psg_ids = retrieval_logits_to_ids(retrieval_logits[0])

                    generated_ids = gen_model.generate(
                        retrieval_ids,
                        max_new_tokens=args.max_new_tokens,
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(retrieval_ids, generated_ids)
                    ]
                elif "e2e" in model_signature and "token" in model_signature:
                    ret_model, gen_model = model

                    # modification for no special token case
                    input_ids_ = model_inputs.input_ids[0].tolist()
                    start_inst_idx = input_ids_.index(1)
                    end_inst_idx = input_ids_.index(4)
                    special_tok_positions = [start_inst_idx]

                    ctx_boundary_span = [781, 29501, 14391, 29515] # find '\n- Title:'

                    if eval_dataset == "nq" or eval_dataset == "hotpotqa":
                        query_boundary_span = [781, 25762, 29515] # find '\nQuestion:'
                    elif eval_dataset == "fever":
                        query_boundary_span = [781, 27922, 29515] # find '\nClaim:'
                    elif eval_dataset == "wow":
                        query_boundary_span = [781, 1624, 26190, 29515] # find '\nConversation:' 

                    indices_of_query = [i for i in range(len(input_ids_) - len(query_boundary_span) + 1) if input_ids_[i:i+len(query_boundary_span)] == query_boundary_span]
                    assert len(indices_of_query) != 1

                    indices_of_ctx = [i for i in range(len(input_ids_) - len(ctx_boundary_span) + 1) if input_ids_[i:i+len(ctx_boundary_span)] == ctx_boundary_span and i < indices_of_query[0]]

                    special_tok_positions += indices_of_ctx
                    special_tok_positions += indices_of_query
                    special_tok_positions.append(end_inst_idx)
                    special_tok_positions = [special_tok_positions]

                    if "blockQuery" in model_signature:
                        # Loading block query attention
                        attention_mask = block_ctx_for_query_attention_mask(model_inputs.input_ids, special_tok_positions, device=model_inputs.input_ids.device)
                    else:
                        attention_mask = None

                    assert len(corpus) == len(indices_of_ctx)

                    retrieval_mask, retrieval_logits = ret_model.forward(
                        input_ids=model_inputs.input_ids, 
                        attention_mask=attention_mask if attention_mask else None,
                        special_tok_positions=special_tok_positions,
                        gumbel_tau=1.0,
                        gumbel_k=args.top_k,
                        )
                    
                    retrieval_ids = torch.masked_select(model_inputs.input_ids, retrieval_mask.bool()).unsqueeze(0).to(torch.long) # num_context, max_context_len, d -> top_k=4, max_context_len, d
                    retrieval_psg_ids = retrieval_logits_to_ids(retrieval_logits[0])

                    generated_ids = gen_model.generate(
                        input_ids=retrieval_ids, 
                        max_new_tokens=args.max_new_tokens,
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(retrieval_ids, generated_ids)
                    ]
                elif args.probe_attention: # Probe attention head for each benchmark's validation set, then using attention head for retrieval
                    
                    eval_flag = True
                    input_ids = model_inputs.input_ids
                    input_ids_ = input_ids.tolist()[0]
                    prompt_enc_states = model(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)

                    special_tok_positions = parse_special_tokens_for_probe_attn(model_inputs=model_inputs, model_signature=model_signature, eval_dataset=eval_dataset, corpus=corpus)
                        
                    retrieved_context = prob_attn_retrieve(model, model_signature, tokenizer, model.config, prompt_enc_states, input_ids[:,-1], args.max_new_tokens, retrieval_score, special_tok_positions, model_inputs.input_ids[0], sorted_ret_head, head_topk=args.head_topk)

                    attention_retrieval.append(retrieved_context)

                    prompt = prompt_template(d, retrieved_context)
                    prompt = sys_prompt_template(prompt)

                    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)

                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        max_new_tokens=args.max_new_tokens,
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]
                elif args.block_rta_context_decoding:
                    if "RTA" not in args.model_signature:
                        print("This decoding is for RTA SFT-ed model only, which blocks RTA's answer generation phase's attention not to access token, but only the retrieval stage tokens.")
                        raise NotImplementedError
                    
                    input_ids = model_inputs.input_ids
                    input_ids_ = input_ids.tolist()[0]
                    
                    prompt_enc_states = model(input_ids=input_ids[:,:-1], use_cache=True, return_dict=True)

                    generated_ids = [decode_with_prefix_states(model, tokenizer, prompt_enc_states, input_ids_, input_ids[:,-1], args.max_new_tokens)]
                else:
                    generated_ids = model.generate(
                        model_inputs.input_ids,
                        attention_mask=None,
                        max_new_tokens=args.max_new_tokens,
                    )

                    generated_ids = [
                        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                    ]

                response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                model_predictions.append(response) # Save model's actual predictions to predictions files
                if "e2e" in model_signature:
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
                    if args.probe_attention:
                        eval_results["retrieval_probe_attention"] = metrics["retrieval_probe_attention"](retrieval_prediction=attention_retrieval, gold_retrieval=gold_retrieval)

            with open(output_dir + "/predictions.txt", "w+", encoding='utf-8') as f:
                f.writelines([p.replace('\n', '').strip()+'\n' for p in model_predictions])

            with open(output_dir + '/eval_result.json', 'w') as f:
                json.dump(eval_results, f, indent=4)

            

        
        
