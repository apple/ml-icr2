#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import argparse

import json
import os

from tqdm import tqdm, trange

import random
# from transformers import AutoModelForCausalLM, AutoTokenizer


def main(args, datasets, models):

    splits = ['train']
    datasets = datasets.split(',')
    models = models.split(',')
    
    random.seed(42)

    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    for split in splits:
        for dataset in datasets:

            print(f"Currently processing {dataset}-{split}....")
            print("Writing to "+f"./icr2_benchmark/{dataset}-{split}-kilt.jsonl")

            # load gold passages
            gold_passages_filename = f"{dataset}-{split}-gold-passage.jsonl"
            with open("./gold_passages/"+gold_passages_filename, "r",encoding="utf-8") as f:
                pos_passages = [json.loads(s.strip()) for s in f.readlines()]
                pos_annotations = {}

                # Overloop each annotation, (query, ans, gold_psg, neg_psgs)
                for d in tqdm(pos_passages):
                    
                    # Skip the instance with no answer / no any passage with associated answer
                    if len(d['answer']) == 0:
                        continue
                    answer_coverage = [len(g_p['associated_answers']) for g_p in d['gold_passages'] if g_p['associated_answers']]
                    if len(answer_coverage) == 0:
                        continue

                    # For NQ, triviaqa we only select one gold passage and its associated answer
                    # We want to select the one with highest answer coverage
                    if dataset == "nq" or dataset == "triviaqa" or dataset == "wow" or dataset == "fever":
                        
                        max_answer_coverage = max(answer_coverage)
                        for g_p in d['gold_passages']:
                            g_p['text'] = g_p['text'].replace("BULLET::::", "")
                            if not g_p['associated_answers']:
                                continue
                            if len(g_p['associated_answers']) == max_answer_coverage:
                                d['gold_passages'] = [g_p] # we select the one gold psg with maximum number of associated answers
                                d['answer'] = g_p['associated_answers']
                                break
                        
                    # Hotpotqa: We need to select all passage for hotpotqa's reasoning
                    elif dataset == "hotpotqa":
                        for g_p in d['gold_passages']:
                            g_p['text'] = g_p['text'].replace("BULLET::::", "")
                            if not g_p['associated_answers']:
                                continue
                        d['gold_passages'] = d['gold_passages']
                        d['answer'] = list(set(d['answer']))

                    else:
                        raise NotImplementedError
                    
                    pos_annotations[d['qid']] = d

            # load negative passages
            # Add negative passages overall retriever
            all_retriever_filtered_predictions = {}
            for model in models:
                negative_passages_filename = f"{dataset}-{split}-kilt-filter.jsonl"
                with open("./filtered_predictions/"+split+'/'+model+'/'+negative_passages_filename, "r",encoding="utf-8") as f:
                    filtered_retriever_predictions = [json.loads(s.strip()) for s in f.readlines()]
                    predictions = {}
                    for d in filtered_retriever_predictions:
                        predictions[d['qid']] = d
                all_retriever_filtered_predictions[model] = predictions

            # From any negative annotaitons, load query set
            negative_passages_filename = f"{dataset}-{split}-kilt-filter.jsonl"
            with open("./filtered_predictions/"+split+'/'+models[0]+'/'+negative_passages_filename, "r",encoding="utf-8") as f:
                valid_query_annotations = [json.loads(s.strip()) for s in f.readlines()]
            
            # Loop each query
            
            id_set = set()

            output_lines = []
            # For each query
            for negative_annotation in tqdm(valid_query_annotations):
                context_corpus = []
                qid = negative_annotation['qid']
                query = negative_annotation['input']

                # Add all positive passages
                for psg in pos_annotations[qid]['gold_passages']:
                    pos_item = {"model": "gold", "title_text": psg['wikipedia_title'], "passage_text": psg['text']}
                    context_corpus.append(pos_item)
                answer = pos_annotations[qid]["answer"]

                # Add negative passages
                len_context = 0
                # length_threshold = 32000 * 0.85
                length_threshold = args.context_length_num_passage # for 32k, we use 200 passages, for 64k, we use 500 passages, for 100K we use 700 passages.
                while len_context < length_threshold and sum([len(all_retriever_filtered_predictions[m][qid]['output']) for m in models]) > 0:
                    for model in models:
                        try:
                            psg = all_retriever_filtered_predictions[model][qid]['output'].pop(0)
                        except:
                            # if model's bucket is empty
                            continue
                        if psg['pid'] not in id_set: # deduplicate
                            id_set.add(psg['pid'])
                        else:
                            continue
                        neg_item = {"pid": psg['pid'], "model": model.replace("_our", ""), "title_text": psg['title_text'], "passage_text": psg['passage_text']}
                        context_corpus.append(neg_item)

                        # Counting context by #tokens, this can be slow
                        # len_context = len(tokenizer.encode(' '.join([d['title']+d['text'] for d in context_corpus])))

                        # Counting context by #psg
                        len_context = len(context_corpus)
                
                
                icr2_data = {}
                icr2_data['qid'] = qid
                icr2_data['query_text'] = query
                icr2_data['answers'] = answer
                icr2_data['corpus'] = context_corpus

                output_lines.append(icr2_data)
            
            if args.context_length_num_passage == 200:
                ctx_length = '32k'
            elif args.context_length_num_passage == 500:
                ctx_length = '64k'
            elif args.context_length_num_passage == 700:
                ctx_length = '1m'
            elif args.context_length_num_passage == 100:
                ctx_length = '16k'

            if split == 'train':
                with open(f"./icr2_benchmark/{dataset}-{ctx_length}-valid-kilt.jsonl", "w+", encoding="utf-8") as f:
                    f.writelines([json.dumps(d)+'\n' for d in output_lines[-100:]])

                with open(f"./icr2_benchmark/{dataset}-{ctx_length}-{split}-kilt.jsonl", "w+", encoding="utf-8") as f:
                    f.writelines([json.dumps(d)+'\n' for d in output_lines[:100]])
            else:
                with open(f"./icr2_benchmark/{dataset}-{ctx_length}-{split}-kilt.jsonl", "w+", encoding="utf-8") as f:
                    f.writelines([json.dumps(d)+'\n' for d in output_lines])
            

    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        default="nq,hotpotqa,wow,fever",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="dpr_our,contriever_our,bm25_our,drqa,blink",
    )

    parser.add_argument(
        "--context-length-num-passage",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    main(
            args=args,
            datasets=args.datasets,
            models=args.models
        )



