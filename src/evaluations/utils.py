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
from collections import defaultdict

import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import torch.nn.functional as F


RETRIEVAL_TOKEN_START = "[RETRIEVAL]"
RETRIEVAL_TOKEN_END = "[/RETRIEVAL]"


def retrieval_response_post_processing(args, eval_dataset, response):

    # For RTA and CCI, we process response to get 1. retrieval item and 2. predicted answer
    if eval_dataset == "nq" or eval_dataset == "hotpotqa" or eval_dataset == "triviaqa" or eval_dataset == "musique" or "f5000-w5000" in args.model_signature: # for in-domain training, we add retrieval metric to wow and fever
        
        

        if eval_dataset == "nq" or eval_dataset == "hotpotqa" or eval_dataset == "musique":
            answer_prefix = "According to these relevant passages, the best answer for the question is:"
        elif eval_dataset == "fever":
            answer_prefix = "According to these relevant passages, the best judgement for the claim is:"
        elif eval_dataset == "wow":
            answer_prefix = "According to these relevant passages, the best completion for the conversation is:"

        if eval_dataset == "nq" or eval_dataset == "hotpotqa" or eval_dataset == "musique":
            cci_retrieval_prefix = "The IDs for all relevant passages to the given query are:"
            rta_retrieval_prefix = "All relevant passages to the given question are:"
        elif eval_dataset == "fever":
            cci_retrieval_prefix = "The IDs for all relevant passages to the given query are:"
            rta_retrieval_prefix = "All relevant passages to the given claim are:"
        elif eval_dataset == "wow":
            cci_retrieval_prefix = "The IDs for all relevant passages to the given query are:"
            rta_retrieval_prefix = "All relevant passages to the given conversation are:"

        # first remove retrieval token
        response = response.strip(RETRIEVAL_TOKEN_END).strip(RETRIEVAL_TOKEN_START)

        if answer_prefix in response:
            predicted_answer = response.split(answer_prefix)[1].strip().strip(".")
            
            retrieval_prediction = response.split(answer_prefix)[0].strip()
            retrieval_prediction = retrieval_prediction.strip(RETRIEVAL_TOKEN_END).strip(RETRIEVAL_TOKEN_START).strip(cci_retrieval_prefix).strip()
            retrieval_prediction = retrieval_prediction.strip(RETRIEVAL_TOKEN_END).strip(RETRIEVAL_TOKEN_START).strip(rta_retrieval_prefix).strip()

        else:
            # If model refuses to follow the instruction and do not answer
            predicted_answer = "I have no comment."
            retrieval_prediction = "I have no comment."

        return retrieval_prediction, predicted_answer
    else:
        # For fever and WOW, using the default response to as predicted answer, and no retrieved prediction so far
        return None, response

def load_loft_gold_retrieval(corpus, queries):
    gold_retrieval = []
    for qid, q in enumerate(queries):
        d = {}
        cids = [d[0] for d in q['metadata']['qrels']]
        d['gold_context_id'], d['gold_context_text'] = [], []

        for cid in cids:
            for pid_in_prompt, ctx in enumerate(corpus):
                if cid == ctx['pid']:
                    d['gold_context_id'].append(str(pid_in_prompt))
                    d['gold_context_text'].append(ctx['passage_text'])

        gold_retrieval.append(d)

    return gold_retrieval

def load_icr2_gold_retrieval(queries):

    gold_retrieval = []
    for qid, q in enumerate(queries):
        # corpus = q['corpus']
        d = {}
        
        d['gold_context_id'], d['gold_context_text'] = [], []
        for i, psg in enumerate(q['corpus']):
            if 'model' in list(psg.keys()):
                if psg['model'] == 'gold':
                    d['gold_context_id'].append(str(i))
                    d['gold_context_text'].append(psg['passage_text'])

        gold_retrieval.append(d)
    
    return gold_retrieval

def retrieval_logits_to_ids(retrieval_logits):
    return [i for i in range(len(retrieval_logits)) if retrieval_logits[i]]

def retrieval_calculate(config, attention_maxtrix, retrieval_score, inp, step_token, needle_start, needle_end , topk=1):
    for layer_idx in range(config.num_hidden_layers):
        for head_idx in range(config.num_attention_heads):
            values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)

            for v, i in zip(values, idx):
                # if  self.needle_start <= i < self.needle_end and inp.item()==self.prompt_ids[i].item():
                if  needle_start <= i < needle_end:
                    # retrieval_score[layer_idx][head_idx][0] += 1/(needle_end - needle_start)
                    retrieval_score[layer_idx][head_idx][0] += 1
                    retrieval_score[layer_idx][head_idx][1] += step_token
                    break

    return retrieval_score


def prob_attn_calculate(model, tokenizer, config, encoding_states, inp, max_dec_len, needle_start, needle_end, retrieval_score):
        output = []
        past_kv = encoding_states.past_key_values
        for step_i in range(max_dec_len):
            inp = inp.view(1, 1)
            outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True, attn_mode="torch" )
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            step_token = tokenizer.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            
            for start, end in zip(needle_start, needle_end):
                retrieval_score = retrieval_calculate(config, outputs.attentions, retrieval_score, inp, step_token, start, end)

            
            if step_token == "</s>": break
        
        return output, retrieval_score

def retrieval_by_attn_head(config, attention_maxtrix, retrieval_score, topk=1, special_tok_positions=None, retrieval_head_idx=None):
    hit = 0
    attn_tok_pos = []
    for layer_idx in range(config.num_hidden_layers):
        for head_idx in range(config.num_attention_heads):
            values, idx = attention_maxtrix[layer_idx][0][head_idx][-1].topk(topk)

            for v, i in zip(values, idx):
                for (ret_head_l_idx, ret_head_h_idx, _) in retrieval_head_idx:
                    if layer_idx == ret_head_l_idx and head_idx == ret_head_h_idx:
                        attn_tok_pos.append(i.item())
    
    passage_range = [find_bin(special_tok_positions, p) for p in attn_tok_pos] 
    retrieved_positions = [torch.arange(d[0], d[1]).cuda() for d in passage_range if d is not None]

    return retrieval_score, hit, retrieved_positions

def prob_attn_retrieve(model, model_signature, tokenizer, config, encoding_states, inp, max_len, retrieval_score, special_tok_positions, input_ids, retrieval_head_idx, head_topk=1):
        output = []
        past_kv = encoding_states.past_key_values
        retrieved_context = []
        
        # if only use 1 token for DA
        # if "DA" in model_signature:
        #     max_len = 1
        
        for step_i in range(max_len): # retrieve on the 1st decoded position
            inp = inp.view(1, 1)
            outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, output_attentions=True, attn_mode="torch" )
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            step_token = tokenizer.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            
            if "DA" in model_signature:
                # For DA, we only use the first decoding step for retrieving.
                retrieval_score, hit, retrieved_positions = retrieval_by_attn_head(config, outputs.attentions, retrieval_score, topk=head_topk, special_tok_positions=special_tok_positions, retrieval_head_idx=retrieval_head_idx)
                retrieved_context += list(set([tokenizer.decode(torch.index_select(input_ids, dim=-1, index=ret)) for ret in retrieved_positions]))
                
            if "RTA" in model_signature or "CCI" in model_signature:
                # For RTA and CCI, we only use the tokens in <RETRIEVAL> and </RETRIEVAL>
                if 32768 in output and 32769 not in output:
                    retrieval_score, hit, retrieved_positions = retrieval_by_attn_head(config, outputs.attentions, retrieval_score, topk=head_topk, special_tok_positions=special_tok_positions, retrieval_head_idx=retrieval_head_idx)
                    retrieved_context += list(set([tokenizer.decode(torch.index_select(input_ids, dim=-1, index=ret)) for ret in retrieved_positions]))
            
            if step_token == "</s>": break

        retrieved_context = list(set(retrieved_context)) # deduplicate
        if "CCI" in model_signature:
            retrieved_outputs = [parse_cci_attn_retrieval_context(s) for s in retrieved_context]
        else:
            retrieved_outputs = [{'title_text': s.strip('\n').split('\n')[0].strip('- Title: '), 'passage_text': s.strip('\n').split('\n')[1]} for s in retrieved_context]

        return retrieved_outputs

def visualize_retrieval_score(retrieval_score, config, dataset_name, save_path):
    
    df = []
    for layer_idx in range(config.num_hidden_layers):
        for head_idx in range(config.num_attention_heads):
            name = 'L'+str(layer_idx)+"_H"+str(head_idx)
            score = retrieval_score[layer_idx][head_idx][0]
            df.append([name, score])
    
    plt.figure(figsize=(50, 3))  # Optional: Set figure size

    df = pd.DataFrame(df, columns=['name', 'score'])
    sns.barplot(x='name', y='score', data=df)

    # Optional: Adding titles and labels
    plt.title('Attention Score on '+dataset_name)
    plt.xlabel('Attn Index')
    plt.ylabel('Score')
    plt.xticks(rotation=90, fontsize=3)

    # Show the plot
    plt.savefig(save_path+'/attn_retrieval_score.pdf', bbox_inches='tight')

def process_retrieval_score(args, retrieval_score, avg_response_len, config, eval_dataset, output_dir, top_k_head):
    print("Total # of decoding step is", avg_response_len)
    retrieval_score_ranklist = []
    for layer_idx in range(config.num_hidden_layers):
        for head_idx in range(config.num_attention_heads):
            retrieval_score[layer_idx][head_idx][0] = retrieval_score[layer_idx][head_idx][0] / avg_response_len
            retrieval_score_ranklist.append((layer_idx, head_idx, retrieval_score[layer_idx][head_idx][0]))
    
    with open(output_dir + "/attn_retrieval_score.txt", "w+", encoding='utf-8') as f:
        f.writelines([str(retrieval_score)])

    visualize_retrieval_score(retrieval_score, config, eval_dataset, output_dir)

    return sorted(retrieval_score_ranklist, key=lambda x: x[2], reverse=True)[:top_k_head] # list of (layer_idx, head_idx, ret_score), sorted by ret_score from large >> small


def load_icr2_dataset(args, eval_dataset, split='dev'):
    # Load ICR2 Datasets
    if 'loft' in args.benchmark_path:
        benchmark_path = './data/icr2_benchmark/'
    else:
        benchmark_path = args.benchmark_path
    dataset_path = "{benchmark_path}/{eval_dataset}-{eval_context_length}-{split}-kilt.jsonl".format(
        benchmark_path = benchmark_path, \
        eval_context_length = args.eval_context_length, \
        eval_dataset = eval_dataset, \
        split=split
    )

    # load queries, answers, retrieved corpus
    queries = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            d = json.loads(line.strip())
            queries.append(d)

    if args.validation:
        iter_ = tqdm(queries[:5])
    else:
        iter_ = tqdm(queries)
            
    references = [d['answers'] for d in queries]

    ### Processing Corpus ###
    if args.gold_needle_mode == "random":
        gold_context_index = []
        for i, q in enumerate(queries):
            random.shuffle(q['corpus'])
            gold_index = []
            for j, d in enumerate(q['corpus']):
                if 'model' in list(d.keys()):
                    if d['model'] == 'gold':
                        gold_index.append(j)
            gold_context_index.append(gold_index)
    else:
        raise NotImplementedError
    
    gold_retrieval = load_icr2_gold_retrieval(queries)

    return iter_, references, gold_retrieval, gold_context_index

def load_icr2_metrics(args, eval_dataset, model_signature):
    # Load Metrics
    metrics = {}
    if eval_dataset == "nq" or eval_dataset == "hotpotqa" or eval_dataset == "triviaqa":
        metrics['em'] = load_metric('em')
        metrics['recallem'] = load_metric('recallem')
        metrics['bidirect_recallem'] = load_metric('bidirect_recallem')
        if "CCI" in model_signature:
            metrics['retrieval_recall_cci'] = load_metric('retrieval_recall_cci')
        elif "RTA" in model_signature:
            metrics['retrieval_recall_rta'] = load_metric('retrieval_recall_rta')
    elif eval_dataset == "fever":
        metrics['em'] = load_metric('em')
        metrics['polarizedem'] = load_metric('polarizedem')
        if "f5000-w5000" in args.model_signature: # for in-domain training, we add retrieval metric to wow and fever
            if "CCI" in model_signature:
                metrics['retrieval_recall_cci'] = load_metric('retrieval_recall_cci')
            elif "RTA" in model_signature:
                metrics['retrieval_recall_rta'] = load_metric('retrieval_recall_rta')
    elif eval_dataset == "wow":
        metrics['bertscore'] = load_metric('bertscore')
        metrics['bleu'] = load_metric('bleu')
        metrics['rouge'] = load_metric('rouge')
        if "f5000-w5000" in args.model_signature: # for in-domain training, we add retrieval metric to wow and fever
            if "CCI" in model_signature:
                metrics['retrieval_recall_cci'] = load_metric('retrieval_recall_cci')
            elif "RTA" in model_signature:
                metrics['retrieval_recall_rta'] = load_metric('retrieval_recall_rta')
    else:
        print("Dataset is not in nq,hotpotqa,triviaqa,fever,wow")
        raise NotImplementedError
    if 'e2e' in model_signature:
        metrics["retrieval_recall_TopK"] = load_metric('retrieval_recall_TopK')
    if args.probe_attention:
        metrics["retrieval_probe_attention"] = load_metric('retrieval_probe_attention')

    return metrics

def parse_input_ids_chunk(model_inputs, eval_dataset, corpus):

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
    assert len(indices_of_query) == 1

    indices_of_ctx = [i for i in range(len(input_ids_) - len(ctx_boundary_span) + 1) if input_ids_[i:i+len(ctx_boundary_span)] == ctx_boundary_span and i < indices_of_query[0]]

    special_tok_positions += indices_of_ctx
    special_tok_positions += indices_of_query
    special_tok_positions.append(end_inst_idx)
    special_tok_positions = [special_tok_positions]
    assert len(corpus) == len(indices_of_ctx)

    return special_tok_positions

def parse_special_tokens_for_probe_attn(model_inputs, model_signature, eval_dataset, corpus):
    # modification for no special token case
    special_tok_positions = []
    input_ids_ = model_inputs.input_ids[0].tolist()

    if "CCI" not in model_signature:
        ctx_boundary_span = [781, 29501, 14391, 29515] # find '\n- Title:\n'
    else:
        ctx_boundary_span = [781, 29501, 8969, 1233, 5287] # find '\n- Passage ID:'

    if eval_dataset == "nq" or eval_dataset == "hotpotqa":
        query_boundary_span = [781, 25762, 29515] # find '\nQuestion:'
    elif eval_dataset == "fever":
        query_boundary_span = [781, 27922, 29515] # find '\nClaim:'
    elif eval_dataset == "wow":
        query_boundary_span = [781, 1624, 26190, 29515] # find '\nConversation:' 

    indices_of_query = [i for i in range(len(input_ids_) - len(query_boundary_span) + 1) if input_ids_[i:i+len(query_boundary_span)] == query_boundary_span]
    assert len(indices_of_query) == 1

    indices_of_ctx = [i for i in range(len(input_ids_) - len(ctx_boundary_span) + 1) if input_ids_[i:i+len(ctx_boundary_span)] == ctx_boundary_span and i < indices_of_query[0]]

    special_tok_positions += indices_of_ctx
    special_tok_positions += indices_of_query

    assert len(corpus) == len(indices_of_ctx)

    return special_tok_positions

def calculate_element_distance(lst, elem1, elem2):
    # Ensure both elements are in the list
    if elem1 in lst and elem2 in lst:
        index1 = lst.index(elem1)
        index2 = lst.index(elem2)
        return abs(index1 - index2)
    else:
        return len(lst)
    
def find_bin(indices, query_int):
    for i in range(len(indices) - 1):
        if indices[i] <= query_int < indices[i + 1]:
            return [indices[i], indices[i + 1]]
    return None

def parse_cci_attn_retrieval_context(text):
    # Split the input text into lines
    lines = text.split('\n')

    # Initialize a dictionary to store the parsed data
    parsed_data = {"title_text": "", "passage_text": ""}

    # Extract the title
    for i, line in enumerate(lines):
        if line.startswith('- Title:'):
            parsed_data['title_text'] = line.split(': ', 1)[1]
            # Extract the text from the line after the title until the end
            parsed_data['passage_text'] = ' '.join(lines[i + 1:]).strip()
            break

    return parsed_data


def block_rta_context_attention_mask(input_ids, special_token_id=32768):
    """
    Generate an attention mask where tokens after a specific token (special_token_id)
    can attend to that token and all tokens after it.
    
    Args:
    - input_ids (list[int]): The input sequence of token IDs.
    - special_token_id (int): The token ID after which attention should be adjusted.
    
    Returns:
    - attention_mask (torch.Tensor): A square attention mask of shape (len(input_ids), len(input_ids)).
    """
    seq_len = len(input_ids)
    attention_mask = torch.zeros(seq_len, seq_len)

    # Find the index of the special token
    if special_token_id in input_ids and special_token_id+1 in input_ids: # only do it when model has retrieved
        special_token_idx = input_ids.index(special_token_id)

        # Create causal attention for all tokens before and including the special token
        attention_mask[:special_token_idx + 1, :special_token_idx + 1] = torch.tril(torch.ones(special_token_idx + 1, special_token_idx + 1))

        # For tokens after the special token, they can attend to the special token and all following tokens
        for i in range(special_token_idx + 1, seq_len):
            attention_mask[i, special_token_idx] = 1.0  # Can attend to the special token
            attention_mask[i, i] = 1.0  # Can attend to itself
            attention_mask[i, special_token_idx + 1:seq_len] = 1.0  # Can attend to all tokens after special token
    else:
        attention_mask = torch.tril(torch.ones(seq_len, seq_len))

    attention_mask = attention_mask.requires_grad_(False)
    # Invert the mask
    attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))  # Set 0s to -inf
    attention_mask = attention_mask.masked_fill(attention_mask == 1, 0.0)  # Set 1s to 0

    return attention_mask


def decode_with_prefix_states(model, tokenizer, encoding_states, sequence_input_ids, inp, max_dec_len):
        output = []
        past_kv = encoding_states.past_key_values

        first_param = next(model.parameters())
        device, dtype = first_param.device, first_param.dtype

        for step_i in range(max_dec_len):
            inp = inp.view(1, 1)
            attention_mask = block_rta_context_attention_mask(sequence_input_ids).to(device).to(dtype)
            attention_mask = attention_mask[-1,:] # get the last 1 mask
            outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, attention_mask=attention_mask, output_attentions=False)
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            sequence_input_ids.append(inp.item())
            step_token = tokenizer.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            
            if step_token == "</s>": break
        
        return output

def decode_with_prefix_states_customized_attn_mask(model, tokenizer, encoding_states, sequence_input_ids, inp, attention_mask, max_dec_len):
        output = []
        past_kv = encoding_states.past_key_values

        first_param = next(model.parameters())
        device, dtype = first_param.device, first_param.dtype
        attention_mask = attention_mask[0][0][-1].unsqueeze(0)

        for step_i in range(max_dec_len):
            inp = inp.view(1, 1)
            
            attention_mask_ = F.pad(attention_mask, (0, len(output)+1), mode='constant', value=0)
            outputs = model(input_ids=inp, past_key_values=past_kv, use_cache=True, attention_mask=attention_mask_, output_attentions=False)
            past_kv = outputs.past_key_values
            inp = outputs.logits[0, -1].argmax()
            sequence_input_ids.append(inp.item())
            step_token = tokenizer.convert_ids_to_tokens(inp.item())
            output.append(inp.item())
            
            if step_token == "</s>": break
        
        return output
