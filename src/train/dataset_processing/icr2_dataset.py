#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import os
import pandas as pd
import json
import sys
sys.path.insert(1, './src') # add src path

from prompt.utils import load_template
from models import build_tokenizer
from transformers import set_seed

from src import prompt
from tqdm import tqdm
import random

from datasets import Value, Features, Sequence, concatenate_datasets
import datasets

import torch
import argparse
import torch.nn.functional as F

# CONTEXT_CHUNK_TOKEN_START = "[CONTEXT]"
# CONTEXT_CHUNK_TOKEN_END = "[/CONTEXT]"

def create_icr2_sft_datset(args, dataset_name, data_size, dataset_read_path, dataset_save_path, model_signature, max_seq_length, masking_prompt, hf_access_token, seed=42, gold_needle_mode="random", save_to_local=False):

    # The __init__ function is run once when instantiating the Dataset object. We 
    # 1) read the dataset file
    # 2) Format the dataset file with prompt template
    # 3) Tokenize input and outptu with model's tokenizer

    set_seed(seed)  # For reproducibility

    max_seq_length = max_seq_length + 1 # for easycontext implementation

    # Parse Model Name
    if "Mistral" in model_signature:
        model_name = "mistral"
    elif "Phi" in model_signature:
        model_name = "phi"
    elif "Qwen" in model_signature:
        model_name = "qwen"
    elif 'Llama' in model_signature:
        model_name = 'llama'
    else:
        raise NotImplementedError

    # Save args to local
    dataset_name_str, dataset_size_str = dataset_name.replace(',' , '-'), data_size.replace(',','-')
    output_folder = f'{dataset_save_path}/{dataset_name_str}-{dataset_size_str}-{model_name}-{args.sft_mode}-hf'
    if masking_prompt:
        output_folder += "compLoss"
    print(f"Saving to HF dataset path {output_folder}...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    argparse_dict = vars(args)
    with open(output_folder+"/icr2_dataset_config.json", "w+") as f:
        json.dump(argparse_dict, f, indent=4)

    all_datasets = dataset_name.split(',')
    all_data_size = [int(size) for size in data_size.split(',')]

    # Load tokenizer
    tokenizer = build_tokenizer(model_signature, model_name=model_name, hf_access_token=hf_access_token)

    hf_datasets = []
    for dataset_i, dataset in enumerate(all_datasets):

        print(f"****************** Currently processing {dataset} with size {all_data_size[dataset_i]} ******************")

        # 1) read the dataset file
        queries = []
        with open(f"{dataset_read_path}/{dataset}-train-kilt.jsonl", 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                d = json.loads(line.strip())
                d['answers'] = list(set(d['answers']))
                queries.append(d)

        random.shuffle(queries)

        # Setting for location of gold passage
        if gold_needle_mode == "random":
            for i, q in enumerate(queries):
                random.shuffle(q['corpus'])
        else:
            raise NotImplementedError

        sys_prompt_template = load_template(model_name+"_sys")
        if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
            prompt_template = load_template('rag-qa')
        elif dataset == "fever":
            prompt_template = load_template('rag-fact-check')
        elif dataset == "wow":
            prompt_template = load_template('rag-dialogue')
        else:
            raise NotImplementedError

        def generator():
            for i, d in enumerate(tqdm(queries[:all_data_size[dataset_i]])):

                if args.sft_mode == "direct-answer":

                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                elif args.sft_mode == "retrieve-then-answer":

                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # First get all gold passages, and process to be prompt format
                    gold_passages = [d for d in d['corpus'] if d['model'] == 'gold']
                    passage_template = load_template('passage')
                    serialized_gold_passages = passage_template.serialize_passages(gold_passages)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    completion = f"All relevant passages to the given query are: \n\n {serialized_gold_passages} \n\n According to these relevant passages, the best response is: {completion}"

                elif args.sft_mode == "cite-context-id":

                    # For citation mark, we ask template to add document id
                    prompt_template.set_document_id()
                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # get gold_passages id
                    gold_passage_ids = [str(idx) for idx, d in enumerate(d['corpus']) if d['model'] == 'gold']
                    gold_passage_ids_str = '[' + ']['.join(gold_passage_ids) +']'

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    completion = f"The IDs for all relevant passages to the given query are: \n\n {gold_passage_ids_str} \n\n According to these relevant passages, the best response is: {completion}"

                else:
                    raise NotImplementedError

                # Tokenize input and output ids
                input_ids = tokenizer([prompt], return_tensors="pt")['input_ids'][0]
                output_ids = tokenizer([completion], return_tensors="pt")['input_ids'][0][1:] # skip bos for completion

                # Concatenate input_ids and output_ids to form as input_ids
                prompt_ids_length = input_ids.shape[-1]
                input_ids = torch.cat([input_ids, output_ids], dim=-1)

                # Replace input_ids's tokens to be PAD to avoid computing loss on prompt
                if masking_prompt:
                    padding_ids = torch.ones(prompt_ids_length) * -100
                    output_ids = torch.cat([padding_ids, output_ids], dim=-1)
                else:
                    output_ids = input_ids

                assert input_ids.shape[-1] == output_ids.shape[-1]
                unpad_seq_length = input_ids.shape[-1]

                # Padding to sequence length
                if unpad_seq_length > max_seq_length:
                    print("unpad_seq_length:", unpad_seq_length, ", is longer than max_seq we set:", max_seq_length)
                    continue

                input_right_pad_until_max_ids = torch.ones(max_seq_length - unpad_seq_length) * tokenizer.convert_tokens_to_ids(tokenizer.eos_token) # Padding input ids with unk
                output_right_pad_until_max_ids = torch.ones(max_seq_length - unpad_seq_length) * -100 # Padding output ids with -100

                output_ids = torch.cat([output_ids, output_right_pad_until_max_ids], dim=-1)
                input_ids = torch.cat([input_ids, input_right_pad_until_max_ids], dim=-1)

                yield {"input_ids": input_ids, "output_ids": output_ids}

        # Specify the sequence length in the feature definition
        features = {"input_ids": Sequence(feature=Value("int64")), "output_ids": Sequence(feature=Value("int64"))}
        hf_datasets.append(datasets.Dataset.from_generator(generator=generator, writer_batch_size=100, features=Features(features)))

        demo_lines = [str(hf_datasets[dataset_i]['input_ids'][0])+'\n', str(hf_datasets[dataset_i]['output_ids'][0])+'\n']
        with open(output_folder+"/"+dataset+"-demo.txt", "w+") as f:
            f.writelines(demo_lines)

    # Concatenate all datasets
    hf_dataset = concatenate_datasets(hf_datasets)

    if save_to_local:
        hf_dataset.save_to_disk(output_folder)

    return hf_dataset


def create_icr2_our_datset(args, dataset_name, data_size, dataset_read_path, dataset_save_path, model_signature, max_seq_length, masking_prompt, hf_access_token, seed=42, gold_needle_mode="random", save_to_local=False):

    # The __init__ function is run once when instantiating the Dataset object. We
    # 1) read the dataset file
    # 2) Format the dataset file with prompt template
    # 3) Tokenize input and outptu with model's tokenizer

    set_seed(seed)  # For reproducibility

    max_seq_length = max_seq_length + 1 # for easycontext implementation

    # Parse Model Name
    if "Mistral" in model_signature:
        model_name = "mistral"
    elif "Phi" in model_signature:
        model_name = "phi"
    elif "Qwen" in model_signature:
        model_name = "qwen"
    else:
        raise NotImplementedError

    # Save args to local
    dataset_name_str, dataset_size_str = dataset_name.replace(',' , '-'), data_size.replace(',','-')
    output_folder = f'{dataset_save_path}/{dataset_name_str}-{dataset_size_str}-{model_name}-{args.sft_mode}-hf'
    if masking_prompt:
        output_folder += "compLoss"
    print(f"Saving to HF dataset path {output_folder}...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    argparse_dict = vars(args)
    with open(output_folder+"/icr2_dataset_config.json", "w+") as f:
        json.dump(argparse_dict, f, indent=4)

    all_datasets = dataset_name.split(',')
    all_data_size = [int(size) for size in data_size.split(',')]

    # Load tokenizer
    tokenizer = build_tokenizer(model_signature, model_name=model_name, hf_access_token=hf_access_token)

    # Add special seperate sign for context chunk (remember to do the same for inference and training)
    # special_tokens_dict = {'additional_special_tokens': [CONTEXT_CHUNK_TOKEN_START, CONTEXT_CHUNK_TOKEN_END]}
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # print(f"{num_added_toks} new tokens have been added...")
    # model.resize_token_embeddings(len(tokenizer))

    hf_datasets = []
    for dataset_i, dataset in enumerate(all_datasets):

        print(f"****************** Currently processing {dataset} with size {all_data_size[dataset_i]} ******************")

        # 1) read the dataset file
        queries = []
        with open(f"{dataset_read_path}/{dataset}-train-kilt.jsonl", 'r', encoding='utf-8') as f:
            invalid_cnt = 0
            for line in tqdm(f.readlines()):
                d = json.loads(line.strip())
                if len(d['corpus']) <= 100:
                    print("Data instance contains passage less than 100!")
                    invalid_cnt+=1
                    continue
                d['answers'] = list(set(d['answers']))
                queries.append(d)
        print("invalid datapoints:", invalid_cnt)
        random.shuffle(queries)

        # Setting for location of gold passage
        if gold_needle_mode == "random":
            for i, q in enumerate(queries):
                random.shuffle(q['corpus'])
        else:
            raise NotImplementedError

        sys_prompt_template = load_template(model_name+"_sys")
        if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
            prompt_template = load_template('rag-qa')
        elif dataset == "fever":
            prompt_template = load_template('rag-fact-check')
        elif dataset == "wow":
            prompt_template = load_template('rag-dialogue')
        else:
            raise NotImplementedError

        # Append [CONTEXT] and concatenate [/CONTEXT] to each context item
        # prompt_template.set_context_special_token()

        def qa_generator():
            for i, d in enumerate(tqdm(queries[:all_data_size[dataset_i]])):

                if args.sft_mode == "direct-answer":

                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                elif args.sft_mode == "retrieve-then-answer":
                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # First get all gold passages, and process to be prompt format
                    gold_passages = [d for d in d['corpus'] if d['model'] == 'gold']
                    passage_template = load_template('passage')
                    serialized_gold_passages = passage_template.serialize_passages(gold_passages)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    completion = f"All relevant passages to the given query are: \n\n {serialized_gold_passages} \n\n According to these relevant passages, the best response is: {completion}"

                elif args.sft_mode == "cite-context-id":

                    raise NotImplementedError

                    # For citation mark, we ask template to add document id
                    prompt_template.set_document_id()
                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # get gold_passages id
                    gold_passage_ids = [str(idx) for idx, d in enumerate(d['corpus']) if d['model'] == 'gold']
                    gold_passage_ids_str = '[' + ']['.join(gold_passage_ids) +']'

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    completion = f"The IDs for all relevant passages to the given query are: \n\n {gold_passage_ids_str} \n\n According to these relevant passages, the best response is: {completion}"

                else:
                    raise NotImplementedError

                valid_flag = True

                # Tokenize input and output ids
                input_ids = tokenizer([prompt], return_tensors="pt")['input_ids'][0]
                output_ids = tokenizer([completion], return_tensors="pt")['input_ids'][0][1:] # skip bos for completion
                # Concatenate input_ids and output_ids to form as input_ids
                input_ids = torch.cat([input_ids, output_ids], dim=-1)

                # # filtering out passage with super long context
                # special_ids = [i for i, (c_eos, c_bos, i_bos, i_eos) in enumerate(zip(
                #                             (input_ids == 32768),
                #                             (input_ids == 32769),
                #                             (input_ids == 1), # using 1 as the starting point rather than 3
                #                             (input_ids == 4)
                #                         )) if c_eos or c_bos or i_bos or i_eos
                #                     ]
                # for i in range(len(special_ids)-1):
                #     ctx_block_length = special_ids[i+1]-special_ids[i]
                #     if ctx_block_length > 700:
                #         print("Context block size is larger than max we set:", 700)
                #         valid_flag = False
                #         break

                input_ids_ = input_ids.tolist()
                start_inst_idx = input_ids_.index(1)
                end_inst_idx = input_ids_.index(4)
                special_tok_positions = [start_inst_idx]

                ctx_boundary_span = [781, 29501, 14391, 29515] # find '\n- Title:'
                query_boundary_span = [781, 25762, 29515] # find '\nQuestion:'

                indices_of_query = [i for i in range(len(input_ids_) - len(query_boundary_span) + 1) if input_ids_[i:i+len(query_boundary_span)] == query_boundary_span]
                if len(indices_of_query) != 1:
                    print('query parsing error, skip this one.')
                    valid_flag = False
                indices_of_ctx = [i for i in range(len(input_ids_) - len(ctx_boundary_span) + 1) if input_ids_[i:i+len(ctx_boundary_span)] == ctx_boundary_span and i < indices_of_query[0]]

                special_tok_positions += indices_of_ctx
                special_tok_positions += indices_of_query
                special_tok_positions.append(end_inst_idx)

                # Padding to sequence length
                if input_ids.shape[-1] + output_ids.shape[-1] > max_seq_length:
                    print("Sequence length is larger than max_seq we set:", max_seq_length)
                    valid_flag = False

                # Check parsed num_psg and num_psg in corpus
                if len(indices_of_ctx) != len(d['corpus']):
                    print("number of indices_of_ctx {d1} does not match with all ctx in gold_psg_ids_list {d2}, skip.".format(d1=len(indices_of_ctx), d2=len(d['corpus'])))
                    valid_flag=False

                gold_psg_ids_list = [pid for pid, p in enumerate(d['corpus']) if p['model'] == "gold"]
                gold_psg_ids = torch.tensor(gold_psg_ids_list, dtype=torch.long)

                gold_psg_one_hot = F.one_hot(gold_psg_ids, num_classes=len(d['corpus']))
                gold_psg_one_hot = gold_psg_one_hot.sum(dim=0).to(dtype=torch.long) # merge multiple correct into one tensor

                if valid_flag:
                    yield {"input_ids": input_ids, "output_ids": output_ids, "gold_psg_one_hot": gold_psg_one_hot, "special_tok_positions": special_tok_positions}
                else:
                    continue

        def factcheck_generator():
            for i, d in enumerate(tqdm(queries[:all_data_size[dataset_i]])):

                if args.sft_mode == "direct-answer":

                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                elif args.sft_mode == "retrieve-then-answer":
                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # First get all gold passages, and process to be prompt format
                    gold_passages = [d for d in d['corpus'] if d['model'] == 'gold']
                    passage_template = load_template('passage')
                    serialized_gold_passages = passage_template.serialize_passages(gold_passages)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    completion = f"All relevant passages to the given query are: \n\n {serialized_gold_passages} \n\n According to these relevant passages, the best response is: {completion}"

                elif args.sft_mode == "cite-context-id":

                    raise NotImplementedError

                    # For citation mark, we ask template to add document id
                    prompt_template.set_document_id()
                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # get gold_passages id
                    gold_passage_ids = [str(idx) for idx, d in enumerate(d['corpus']) if d['model'] == 'gold']
                    gold_passage_ids_str = '[' + ']['.join(gold_passage_ids) +']'

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    completion = f"The IDs for all relevant passages to the given query are: \n\n {gold_passage_ids_str} \n\n According to these relevant passages, the best response is: {completion}"

                else:
                    raise NotImplementedError

                valid_flag = True

                # Tokenize input and output ids
                input_ids = tokenizer([prompt], return_tensors="pt")['input_ids'][0]
                output_ids = tokenizer([completion], return_tensors="pt")['input_ids'][0][1:] # skip bos for completion
                # Concatenate input_ids and output_ids to form as input_ids
                input_ids = torch.cat([input_ids, output_ids], dim=-1)

                # # filtering out passage with super long context
                # special_ids = [i for i, (c_eos, c_bos, i_bos, i_eos) in enumerate(zip(
                #                             (input_ids == 32768),
                #                             (input_ids == 32769),
                #                             (input_ids == 1), # using 1 as the starting point rather than 3
                #                             (input_ids == 4)
                #                         )) if c_eos or c_bos or i_bos or i_eos
                #                     ]
                # for i in range(len(special_ids)-1):
                #     ctx_block_length = special_ids[i+1]-special_ids[i]
                #     if ctx_block_length > 700:
                #         print("Context block size is larger than max we set:", 700)
                #         valid_flag = False
                #         break

                input_ids_ = input_ids.tolist()
                start_inst_idx = input_ids_.index(1)
                end_inst_idx = input_ids_.index(4)
                special_tok_positions = [start_inst_idx]

                ctx_boundary_span = [781, 29501, 14391, 29515] # find '\n- Title:'
                query_boundary_span = [781, 27922, 29515] # find '\nClaim:'

                indices_of_query = [i for i in range(len(input_ids_) - len(query_boundary_span) + 1) if input_ids_[i:i+len(query_boundary_span)] == query_boundary_span]
                if len(indices_of_query) != 1:
                    print('query parsing error, skip this one.')
                    valid_flag = False
                indices_of_ctx = [i for i in range(len(input_ids_) - len(ctx_boundary_span) + 1) if input_ids_[i:i+len(ctx_boundary_span)] == ctx_boundary_span and i < indices_of_query[0]]

                special_tok_positions += indices_of_ctx
                special_tok_positions += indices_of_query
                special_tok_positions.append(end_inst_idx)

                # Padding to sequence length
                if input_ids.shape[-1] + output_ids.shape[-1] > max_seq_length:
                    print("Sequence length is larger than max_seq we set:", max_seq_length)
                    valid_flag = False

                # Check parsed num_psg and num_psg in corpus
                if len(indices_of_ctx) != len(d['corpus']):
                    print("number of indices_of_ctx {d1} does not match with all ctx in gold_psg_ids_list {d2}, skip.".format(d1=len(indices_of_ctx), d2=len(d['corpus'])))
                    valid_flag=False

                gold_psg_ids_list = [pid for pid, p in enumerate(d['corpus']) if p['model'] == "gold"]
                gold_psg_ids = torch.tensor(gold_psg_ids_list, dtype=torch.long)

                gold_psg_one_hot = F.one_hot(gold_psg_ids, num_classes=len(d['corpus']))
                gold_psg_one_hot = gold_psg_one_hot.sum(dim=0).to(dtype=torch.long) # merge multiple correct into one tensor

                if valid_flag:
                    yield {"input_ids": input_ids, "output_ids": output_ids, "gold_psg_one_hot": gold_psg_one_hot, "special_tok_positions": special_tok_positions}
                else:
                    continue

        def dialog_generator():
            for i, d in enumerate(tqdm(queries[:all_data_size[dataset_i]])):

                if args.sft_mode == "direct-answer":

                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                elif args.sft_mode == "retrieve-then-answer":
                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # First get all gold passages, and process to be prompt format
                    gold_passages = [d for d in d['corpus'] if d['model'] == 'gold']
                    passage_template = load_template('passage')
                    serialized_gold_passages = passage_template.serialize_passages(gold_passages)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    completion = f"All relevant passages to the given query are: \n\n {serialized_gold_passages} \n\n According to these relevant passages, the best response is: {completion}"

                elif args.sft_mode == "cite-context-id":

                    raise NotImplementedError

                    # For citation mark, we ask template to add document id
                    prompt_template.set_document_id()
                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # get gold_passages id
                    gold_passage_ids = [str(idx) for idx, d in enumerate(d['corpus']) if d['model'] == 'gold']
                    gold_passage_ids_str = '[' + ']['.join(gold_passage_ids) +']'

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    completion = f"The IDs for all relevant passages to the given query are: \n\n {gold_passage_ids_str} \n\n According to these relevant passages, the best response is: {completion}"

                else:
                    raise NotImplementedError

                valid_flag = True

                # Tokenize input and output ids
                input_ids = tokenizer([prompt], return_tensors="pt")['input_ids'][0]
                output_ids = tokenizer([completion], return_tensors="pt")['input_ids'][0][1:] # skip bos for completion
                # Concatenate input_ids and output_ids to form as input_ids
                input_ids = torch.cat([input_ids, output_ids], dim=-1)

                # # filtering out passage with super long context
                # special_ids = [i for i, (c_eos, c_bos, i_bos, i_eos) in enumerate(zip(
                #                             (input_ids == 32768),
                #                             (input_ids == 32769),
                #                             (input_ids == 1), # using 1 as the starting point rather than 3
                #                             (input_ids == 4)
                #                         )) if c_eos or c_bos or i_bos or i_eos
                #                     ]
                # for i in range(len(special_ids)-1):
                #     ctx_block_length = special_ids[i+1]-special_ids[i]
                #     if ctx_block_length > 700:
                #         print("Context block size is larger than max we set:", 700)
                #         valid_flag = False
                #         break

                input_ids_ = input_ids.tolist()
                start_inst_idx = input_ids_.index(1)
                end_inst_idx = input_ids_.index(4)
                special_tok_positions = [start_inst_idx]

                ctx_boundary_span = [781, 29501, 14391, 29515] # find '\n- Title:'
                query_boundary_span = [781, 1624, 26190, 29515] # find '\nConversation:'


                indices_of_query = [i for i in range(len(input_ids_) - len(query_boundary_span) + 1) if input_ids_[i:i+len(query_boundary_span)] == query_boundary_span]
                if len(indices_of_query) != 1:
                    print('query parsing error, skip this one.')
                    valid_flag = False
                indices_of_ctx = [i for i in range(len(input_ids_) - len(ctx_boundary_span) + 1) if input_ids_[i:i+len(ctx_boundary_span)] == ctx_boundary_span and i < indices_of_query[0]]

                special_tok_positions += indices_of_ctx
                special_tok_positions += indices_of_query
                special_tok_positions.append(end_inst_idx)

                # Padding to sequence length
                if input_ids.shape[-1] + output_ids.shape[-1] > max_seq_length:
                    print("Sequence length is larger than max_seq we set:", max_seq_length)
                    valid_flag = False

                # Check parsed num_psg and num_psg in corpus
                if len(indices_of_ctx) != len(d['corpus']):
                    print("number of indices_of_ctx {d1} does not match with all ctx in gold_psg_ids_list {d2}, skip.".format(d1=len(indices_of_ctx), d2=len(d['corpus'])))
                    valid_flag=False

                gold_psg_ids_list = [pid for pid, p in enumerate(d['corpus']) if p['model'] == "gold"]
                gold_psg_ids = torch.tensor(gold_psg_ids_list, dtype=torch.long)

                gold_psg_one_hot = F.one_hot(gold_psg_ids, num_classes=len(d['corpus']))
                gold_psg_one_hot = gold_psg_one_hot.sum(dim=0).to(dtype=torch.long) # merge multiple correct into one tensor

                if valid_flag:
                    yield {"input_ids": input_ids, "output_ids": output_ids, "gold_psg_one_hot": gold_psg_one_hot, "special_tok_positions": special_tok_positions}
                else:
                    continue


        # Specify the sequence length in the feature definition
        features = {"input_ids": Sequence(feature=Value("int64")), "output_ids": Sequence(feature=Value("int64")), "gold_psg_one_hot": Sequence(feature=Value("int64")), "special_tok_positions": Sequence(feature=Value("int64"))}

        if dataset == "nq" or dataset == 'hotpotqa':
            hf_datasets.append(datasets.Dataset.from_generator(generator=qa_generator, writer_batch_size=100, features=Features(features)))
        elif dataset == 'fever':
            hf_datasets.append(datasets.Dataset.from_generator(generator=factcheck_generator, writer_batch_size=100, features=Features(features)))
        elif dataset == 'wow':
            hf_datasets.append(datasets.Dataset.from_generator(generator=dialog_generator, writer_batch_size=100, features=Features(features)))

        demo_lines = [str(hf_datasets[dataset_i]['input_ids'][0])+'\n', str(hf_datasets[dataset_i]['output_ids'][0])+'\n', str(hf_datasets[dataset_i]['gold_psg_one_hot'][0])+'\n', str(hf_datasets[dataset_i]['special_tok_positions'][0])+'\n']
        with open(output_folder+"/"+dataset+"-demo.txt", "w+") as f:
            f.writelines(demo_lines)

    # Concatenate all datasets
    hf_dataset = concatenate_datasets(hf_datasets)

    if save_to_local:
        hf_dataset.save_to_disk(output_folder)

    return hf_dataset


RETRIEVAL_TOKEN_START = "[RETRIEVAL]"
RETRIEVAL_TOKEN_END = "[/RETRIEVAL]"

def create_icr2_sft_datset_v2(args, dataset_name, data_size, dataset_read_path, dataset_save_path, model_signature, max_seq_length, masking_prompt, hf_access_token, seed=42, gold_needle_mode="random", save_to_local=False):

    # The __init__ function is run once when instantiating the Dataset object. We
    # 1) read the dataset file
    # 2) Format the dataset file with prompt template
    # 3) Tokenize input and outptu with model's tokenizer

    set_seed(seed)  # For reproducibility

    max_seq_length = max_seq_length + 1 # for easycontext implementation

    # Parse Model Name
    if "Mistral" in model_signature:
        model_name = "mistral"
    elif "Phi" in model_signature:
        model_name = "phi"
    elif "Qwen" in model_signature:
        model_name = "qwen"
    elif 'Llama' in model_signature:
        model_name = 'llama'
    else:
        raise NotImplementedError

    # Save args to local
    dataset_name_str, dataset_size_str = dataset_name.replace(',' , '-'), data_size.replace(',','-')
    output_folder = f'{dataset_save_path}/{dataset_name_str}-{dataset_size_str}-{model_name}-{args.sft_mode}-hf-v2'
    if masking_prompt:
        output_folder += "compLoss"
    print(f"Saving to HF dataset path {output_folder}...")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    argparse_dict = vars(args)
    with open(output_folder+"/icr2_dataset_config.json", "w+") as f:
        json.dump(argparse_dict, f, indent=4)

    all_datasets = dataset_name.split(',')
    all_data_size = [int(size) for size in data_size.split(',')]

    # Load tokenizer
    tokenizer = build_tokenizer(model_signature, model_name=model_name, hf_access_token=hf_access_token)

    # For V2 SFT, we add special token for model to perform retrieval
    # Reshape model's embedding to add special tokens
    special_tokens_dict = {'additional_special_tokens': [RETRIEVAL_TOKEN_START, RETRIEVAL_TOKEN_END]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"{num_added_toks} new tokens have been added...:", special_tokens_dict)

    hf_datasets = []
    for dataset_i, dataset in enumerate(all_datasets):

        print(f"****************** Currently processing {dataset} with size {all_data_size[dataset_i]} ******************")

        # 1) read the dataset file
        queries = []
        with open(f"{dataset_read_path}/{dataset}-train-kilt.jsonl", 'r', encoding='utf-8') as f:
            for line in tqdm(f.readlines()):
                d = json.loads(line.strip())
                d['answers'] = list(set(d['answers']))
                queries.append(d)

        random.shuffle(queries)

        # Setting for location of gold passage
        if gold_needle_mode == "random":
            for i, q in enumerate(queries):
                random.shuffle(q['corpus'])
        else:
            raise NotImplementedError

        sys_prompt_template = load_template(model_name+"_sys")
        if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
            prompt_template = load_template('rag-qa')
        elif dataset == "fever":
            prompt_template = load_template('rag-fact-check')
        elif dataset == "wow":
            prompt_template = load_template('rag-dialogue')
        else:
            raise NotImplementedError

        def generator():
            for i, d in enumerate(tqdm(queries[:all_data_size[dataset_i]])):

                valid_flag = True

                if args.sft_mode == "direct-answer":

                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                elif args.sft_mode == "retrieve-then-answer":

                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # First get all gold passages, and process to be prompt format
                    gold_passages = [d for d in d['corpus'] if d['model'] == 'gold']
                    passage_template = load_template('passage')
                    serialized_gold_passages = passage_template.serialize_passages(gold_passages)

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    if dataset == "nq" or dataset == "hotpotqa":
                        completion = f"{RETRIEVAL_TOKEN_START}All relevant passages to the given question are: \n\n {serialized_gold_passages} {RETRIEVAL_TOKEN_END}\n\n According to these relevant passages, the best answer for the question is: {completion}"
                    elif dataset == "fever":
                        completion = f"{RETRIEVAL_TOKEN_START}All relevant passages to the given claim are: \n\n {serialized_gold_passages} {RETRIEVAL_TOKEN_END}\n\n According to these relevant passages, the best judgement for the claim is: {completion}"
                    elif dataset == "wow":
                        completion = f"{RETRIEVAL_TOKEN_START}All relevant passages to the given conversation are: \n\n {serialized_gold_passages} {RETRIEVAL_TOKEN_END}\n\n According to these relevant passages, the best completion for the conversation is: {completion}"

                elif args.sft_mode == "cite-context-id":

                    # For citation mark, we ask template to add document id
                    prompt_template.set_document_id()
                    prompt = prompt_template(d, d['corpus'])
                    prompt = sys_prompt_template(prompt)

                    # get gold_passages id
                    gold_passage_ids = [str(idx) for idx, d in enumerate(d['corpus']) if d['model'] == 'gold']
                    gold_passage_ids_str = '[' + ']['.join(gold_passage_ids) +']'

                    if dataset == "nq" or dataset == "hotpotqa" or dataset == "triviaqa":
                        if len(d['answers']) == 1:
                            completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                        else:
                            completion = "There are multiple correct answers. All possible answers for this question are " + ', '.join([f"({i}) {ans}" for i, ans in enumerate(d['answers'])]) + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "fever":
                        completion = d['answers'][0] + "." + tokenizer.eos_token # append eos for completion
                    elif dataset == "wow":
                        completion = d['answers'][0] + tokenizer.eos_token # append eos for completion
                    else:
                        raise NotImplementedError

                    # completion = f"{RETRIEVAL_TOKEN_START}The IDs for all relevant passages to the given query are: \n\n {gold_passage_ids_str} {RETRIEVAL_TOKEN_END}\n\n According to these relevant passages, the best response is: {completion}"

                    if dataset == "nq" or dataset == "hotpotqa":
                        completion = f"{RETRIEVAL_TOKEN_START}The IDs for all relevant passages to the given query are: \n\n {gold_passage_ids_str} {RETRIEVAL_TOKEN_END}\n\n According to these relevant passages, the best answer for the question is: {completion}"
                    elif dataset == "fever":
                        completion = f"{RETRIEVAL_TOKEN_START}The IDs for all relevant passages to the given query are: \n\n {gold_passage_ids_str} {RETRIEVAL_TOKEN_END}\n\n According to these relevant passages, the best judgement for the claim is: {completion}"
                    elif dataset == "wow":
                        completion = f"{RETRIEVAL_TOKEN_START}The IDs for all relevant passages to the given query are: \n\n {gold_passage_ids_str} {RETRIEVAL_TOKEN_END}\n\n According to these relevant passages, the best completion for the conversation is: {completion}"

                else:
                    raise NotImplementedError

                # Tokenize input and output ids
                input_ids = tokenizer([prompt], return_tensors="pt")['input_ids'][0]
                output_ids = tokenizer([completion], return_tensors="pt")['input_ids'][0][1:] # skip bos for completion

                ## Parsing to get context block
                if model_name == 'mistral':
                    input_ids_ = input_ids.tolist()
                    start_inst_idx = input_ids_.index(1)
                    end_inst_idx = input_ids_.index(4)
                    special_tok_positions = [start_inst_idx]

                    if dataset == "nq" or dataset == "hotpotqa":
                        ctx_boundary_span = [781, 29501, 14391, 29515] # find '\n- Title:'
                        query_boundary_span = [781, 25762, 29515] # find '\nQuestion:'
                    elif dataset == "fever":
                        ctx_boundary_span = [781, 29501, 14391, 29515] # find '\n- Title:'
                        query_boundary_span = [781, 27922, 29515] # find '\nClaim:'
                    elif dataset == "wow":
                        ctx_boundary_span = [781, 29501, 14391, 29515] # find '\n- Title:'
                        query_boundary_span = [781, 1624, 26190, 29515] # find '\nConversation:'

                elif model_name == "llama":

                    input_ids_ = input_ids.tolist()
                    start_inst_idx = input_ids_.index(tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)[0])
                    end_inst_idx = input_ids_.index(tokenizer.encode("<|eot_id|>", add_special_tokens=False)[0])
                    special_tok_positions = [start_inst_idx]

                    ctx_boundary_span = tokenizer.encode('\n- Title:', add_special_tokens=False)[1:] # find '\n- Title:'
                    if dataset == "nq" or dataset == "hotpotqa":
                        query_boundary_span = tokenizer.encode('\nQuestion:', add_special_tokens=False)[1:] # find '\nQuestion:'
                    elif dataset == "fever":
                        query_boundary_span = tokenizer.encode('\nClaim:', add_special_tokens=False)[1:] # find '\nClaim:'
                    elif dataset == "wow":
                        query_boundary_span = tokenizer.encode('\nConversation:\n', add_special_tokens=False)[1:] # find '\nConversation:\n'

                indices_of_query = [i for i in range(len(input_ids_) - len(query_boundary_span) + 1) if input_ids_[i:i+len(query_boundary_span)] == query_boundary_span]
                if len(indices_of_query) != 1:
                    print('query parsing error, skip this one.')
                    valid_flag = False
                indices_of_ctx = [i for i in range(len(input_ids_) - len(ctx_boundary_span) + 1) if input_ids_[i:i+len(ctx_boundary_span)] == ctx_boundary_span and i < indices_of_query[0]]

                special_tok_positions += indices_of_ctx
                special_tok_positions += indices_of_query
                special_tok_positions.append(end_inst_idx)

                # Concatenate input_ids and output_ids to form as input_ids
                prompt_ids_length = input_ids.shape[-1]
                input_ids = torch.cat([input_ids, output_ids], dim=-1)

                # Replace input_ids's tokens to be PAD to avoid computing loss on prompt
                if masking_prompt:
                    padding_ids = torch.ones(prompt_ids_length) * -100
                    output_ids = torch.cat([padding_ids, output_ids], dim=-1)
                else:
                    output_ids = input_ids

                assert input_ids.shape[-1] == output_ids.shape[-1]
                unpad_seq_length = input_ids.shape[-1]

                # Padding to sequence length
                if unpad_seq_length > max_seq_length:
                    print("unpad_seq_length:", unpad_seq_length, ", is longer than max_seq we set:", max_seq_length)
                    valid_flag = False

                # input_right_pad_until_max_ids = torch.ones(max_seq_length - unpad_seq_length) * tokenizer.convert_tokens_to_ids(tokenizer.unk_token) # Padding input ids with unk
                # output_right_pad_until_max_ids = torch.ones(max_seq_length - unpad_seq_length) * -100 # Padding output ids with -100

                output_ids = torch.cat([output_ids], dim=-1)
                input_ids = torch.cat([input_ids], dim=-1)

                if valid_flag:
                    yield {"input_ids": input_ids, "output_ids": output_ids, "special_tok_positions": special_tok_positions}
                else:
                    continue

        # Specify the sequence length in the feature definition
        features = {"input_ids": Sequence(feature=Value("int64")), "output_ids": Sequence(feature=Value("int64")), "special_tok_positions": Sequence(feature=Value("int64"))}
        hf_datasets.append(datasets.Dataset.from_generator(generator=generator, writer_batch_size=100, features=Features(features)))

        demo_lines = [str(hf_datasets[dataset_i]['input_ids'][0])+'\n', str(hf_datasets[dataset_i]['output_ids'][0])+'\n', str(hf_datasets[dataset_i]['special_tok_positions'][0])+'\n']
        with open(output_folder+"/"+dataset+"-demo.txt", "w+") as f:
            f.writelines(demo_lines)

    # Concatenate all datasets
    hf_dataset = concatenate_datasets(hf_datasets)

    if save_to_local:
        hf_dataset.save_to_disk(output_folder)

    return hf_dataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        default="nq,hotpotqa",
    )

    parser.add_argument(
        "--data-size",
        type=str,
        default="5000,5000",
    )

    parser.add_argument(
        "--model-signature",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )

    parser.add_argument(
        "--dataset-read-path",
        type=str,
        default="data/icr2_benchmark/",
    )

    parser.add_argument(
        "--dataset-save-path",
        type=str,
        default="data/icr2_train_data/",
    )

    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=32768,
    )

    parser.add_argument("--completion-loss-only", action="store_true", help="set True to only backprop loss from the completion; else we also calculate prompt's loss")

    parser.add_argument(
        "--sft-mode",
        type=str,
        default="direct-answer",
        choices=['direct-answer', 'retrieve-then-answer', 'cite-context-id']
    )

    parser.add_argument(
        "--dataset-type",
        type=str,
        default="icr2-ours",
        choices=['icr2-ours', 'sft', 'sft-v2']
    )

    args = parser.parse_args()
    argparse_dict = vars(args)
    with open("icr2_dataset_config.json", "w+") as f:
        json.dump(argparse_dict, f, indent=4)

    if args.dataset_type == "sft":
        dataset = create_icr2_sft_datset(args=args, \
                                         dataset_name=args.datasets, \
                                         data_size=args.data_size, \
                                         dataset_read_path=args.dataset_read_path, \
                                         dataset_save_path=args.dataset_save_path, \
                                         model_signature=args.model_signature, \
                                         hf_access_token="<hf_token>", \
                                         save_to_local=True, \
                                         max_seq_length=args.max_seq_length, \
                                         masking_prompt=args.completion_loss_only \
                                         )
    elif args.dataset_type == "sft-v2":
        dataset = create_icr2_sft_datset_v2(args=args, \
                                            dataset_name=args.datasets, \
                                            data_size=args.data_size, \
                                            dataset_read_path=args.dataset_read_path, \
                                            dataset_save_path=args.dataset_save_path, \
                                            model_signature=args.model_signature, \
                                            hf_access_token="<hf_token>", \
                                            save_to_local=True, \
                                            max_seq_length=args.max_seq_length, \
                                            masking_prompt=args.completion_loss_only \
                                            )
    elif args.dataset_type == "icr2-ours":
        dataset = create_icr2_our_datset(args=args, \
                                         dataset_name=args.datasets, \
                                         data_size=args.data_size, \
                                         dataset_read_path=args.dataset_read_path, \
                                         dataset_save_path=args.dataset_save_path, \
                                         model_signature=args.model_signature, \
                                         hf_access_token="<hf_token>", \
                                         save_to_local=True, \
                                         max_seq_length=args.max_seq_length, \
                                         masking_prompt=False \
                                         )
