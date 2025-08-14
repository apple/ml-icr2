#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import multiprocessing
import json
import re
import string
from tqdm import tqdm
from collections import defaultdict

import random
from collections import defaultdict
from fast_bleu import BLEU, SelfBLEU



def load_passage_corpus(filename="./corpus/kilt_w100_title_our.jsonl"):
    corpus = defaultdict(dict)
    print("Start loading PASSAGE corpus...")
    with open(filename, "r", encoding="utf-8") as f:
        for s in tqdm(f.readlines()):
            d = json.loads(s.strip())
            content = d['contents']
            assert len(content.split('\n')) == 2
            title, text = content.split('\n')

            out_d = {}
            out_d["pid"] = d['id']
            out_d['title_text'] = title
            out_d['passage_text'] = text

            corpus[out_d['pid']] = out_d

    return corpus

psg_corpus = load_passage_corpus()


def load_annotation_data(dataset, split):
    gold_document_annotation = {}
    with open("./our_data/"+dataset+"-"+split+"-kilt.jsonl", "r") as f:
        df = [json.loads(d.strip()) for d in f.readlines()]
    for d in df:
        gold_document_annotation[d['id']] = d

    return gold_document_annotation
    


def load_document_corpus(filename="./corpus/kilt_document_corpus.json"):
    
    print("Start loading DOCUMENT corpus...")
    with open(filename, "r",encoding="utf-8") as f:
        corpus = json.load(f)
    return corpus

doc_corpus = load_document_corpus()

def write_retrieval_predictions(list_of_dict, filename):
    with open(filename, "w+", encoding="utf-8") as f:
        f.writelines([json.dumps(d)+'\n' for d in list_of_dict])

def filtering_provenance_with_bleu(provenance, gold_texts):

    ## Usage:
    ## provenance: list of dict: {pid:, title_text:, passage_text:}
    ## gold_texts: list of string

    # Tokenize the gold text and retrieved passages
    gold_texts = [txt.split(' ') for txt in gold_texts]
    provenance_text = [prov['passage_text'].split(' ') for prov in provenance]
    
    weights = {'bigram': (1/2., 1/2.), 'trigram': (1/3., 1/3., 1/3.)}
    
    # FastBLEU: Compare all provenance_text to a fix set of gold_texts
    bleu = BLEU(gold_texts, weights)
    bigram_bleu = bleu.get_score(provenance_text)['bigram'] # [1.0, 0.0]
    
    filtered_provenance = []
    for i, similarity in enumerate(bigram_bleu):
        if similarity > 0.9:
            continue
        filtered_provenance.append(provenance[i])
    
    return filtered_provenance

def _normalize_text(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    text = text.lower()
    text = "".join(char for char in text if char not in set(string.punctuation))
    text = re.sub(regex, " ", text)
    text = " ".join(text.split())
    return text

def filtering_provenance_with_gold_ans_EM(provenance, gold_answer_list):
    ## Usage:
    ## provenance: list of dict: {pid:, title_text:, passage_text:}
    ## gold_texts: list of string

    filtered_provenance = []
    for prov in provenance:
        annot = True
        provenance_text = prov['passage_text']
        provenance_text = _normalize_text(provenance_text)
        for gold_answer in gold_answer_list:
            gold_answer = _normalize_text(gold_answer)
            if gold_answer in provenance_text:
                annot = False
                break
        if annot:
            filtered_provenance.append(prov)
    
    return filtered_provenance
        

def filter_retrieval_predictions(models, dataset, split, gold_passage_collection, psg_corpus):

    for model in models:
        print("Currently processing model:", model)
        prediction_file_path = "./predictions/"+split+"/"+model+"/"+dataset+"-"+split+"-kilt.jsonl"
        with open(prediction_file_path, "r",encoding="utf-8") as f:
            retrieval_predictions = [json.loads(s.strip()) for s in f.readlines()]

        if model == "dpr_our" or model == "contriever_our" or model == "bm25_our":
            
            filter_negative_retrieval_predictions = []
            for pred in tqdm(retrieval_predictions):
                qid = pred["id"]
                
                # Skip the not selected qeury in dev
                # Or skip if there is no gold passage
                if qid not in list(gold_passage_collection.keys()):
                    continue
                
                # query
                input = pred['input']
                provenance_list = pred['output'][0]['provenance']

                # Getting the string of provenance
                provenance = [psg_corpus[prov['title']] for prov in provenance_list]

                # Getting the gold text
                gold_texts, gold_answers = gold_passage_collection[qid]
                
                filtered_provenance = filtering_provenance_with_bleu(provenance, gold_texts)
                gold_answers = list(set(gold_answers))
                filtered_provenance = filtering_provenance_with_gold_ans_EM(filtered_provenance, gold_answers)
                
                filter_negative_retrieval_predictions.append({"qid": qid, "input": input, "output": filtered_provenance})
            
            write_retrieval_predictions(filter_negative_retrieval_predictions, "./filtered_predictions/"+split+"/"+model+"/"+dataset+"-"+split+"-kilt-filter.jsonl")
        
        elif model == "blink" or model == "drqa":
            
            gold_annotation = load_annotation_data(dataset, split)

            filter_negative_retrieval_predictions = []

            ## FOR EACH QUERY
            for pred in tqdm(retrieval_predictions):
                
                qid = pred["id"]

                # Skip the not selected qeury in dev
                # Or skip if there is no gold passage
                if qid not in list(gold_passage_collection.keys()):
                    continue
                
                # From gold annotation, load a dict transforms docid to all relevant paragraph id
                relevant_docid2paraid = defaultdict(list)
                gold_answers = []
                for output in gold_annotation[qid]['output']:
                    # Fix here, we add all provenance as gold to avoid adding to context
                    if "answer" in list(output.keys()):
                        gold_answers.append(output['answer'])
                    if "provenance" in list(output.keys()):
                        for p in output['provenance']:
                            start_paragraph_id, end_paragraph_id = p['start_paragraph_id'], p['end_paragraph_id']
                            relevant_docid2paraid[p['wikipedia_id']] += [i for i in range(start_paragraph_id, end_paragraph_id+1)]    
                    else:
                        continue
                
                # getting query and retrieval documents
                input = pred['input']
                provenance_list = pred['output'][0]['provenance']

                # Removing gold paragraphs from retrieved document
                all_negative_paragraphs = []

                # FOR EACH PROVENANCE DOCUMENT
                if model == "drqa": # control doc relevance to Top10 documents
                    provenance_list[:10]

                for prov in provenance_list: 
                    doc_id = prov['wikipedia_id']
                    try:
                        document = doc_corpus[doc_id]
                    except:
                        continue

                    gold_paragraphs_ids = relevant_docid2paraid[doc_id]
                    
                    # Add negative_paragraphs from document
                    negative_paragraphs = []
                    for pid, p in enumerate(document['text']):
                        p_d = {}

                        # skip title paragraph, or paragraph with positive pid, or paragraph contains Section label
                        if (pid == 0) or (pid in gold_paragraphs_ids) or ("Section::::" in p):
                            continue
                        if pid not in gold_paragraphs_ids:
                            # Clean p to be the same as gold passsage
                            p = p.replace('BULLET::::', "").replace("\n", "")

                            p_d['pid'] = f"{doc_id}-{pid}"
                            p_d['title_text'] = document['wikipedia_title']
                            p_d['passage_text'] = p
                            negative_paragraphs.append(p_d)
                    
                    all_negative_paragraphs += negative_paragraphs
                    
                random.shuffle(all_negative_paragraphs)
                all_negative_paragraphs = all_negative_paragraphs[:400]

                # Fix here, we use EM filtering for DRQA and BLINK as well
                gold_answers = set(list(gold_answers))
                all_negative_paragraphs = filtering_provenance_with_gold_ans_EM(all_negative_paragraphs, gold_answers)
                
                filter_negative_retrieval_predictions.append({"qid": qid, "input": input, "output": all_negative_paragraphs})
            
            write_retrieval_predictions(filter_negative_retrieval_predictions, "./filtered_predictions/"+split+"/"+model+"/"+dataset+"-"+split+"-kilt-filter.jsonl")
        
        else:
            raise NotImplementedError
        
        

    return None

def main(datasets, models):

    splits = ['dev']
    datasets = datasets.split(',')
    models = models.split(',')


    # doc_corpus = load_document_corpus()

    for split in splits:
        for dataset in datasets:

            # load gold passages
            gold_passages_filename = f"{dataset}-{split}-gold-passage.jsonl"
            with open("./gold_passages/"+gold_passages_filename, "r",encoding="utf-8") as f:
                raw_gold_passages = [json.loads(s.strip()) for s in f.readlines()]
            
            random.seed(42)
            
            # Note that some query with no answer because answer is annotated in the title
            gold_passages = []
            for i, gold in enumerate(raw_gold_passages):
                
                # Filter out the query with no gold_passages/answer
                if len(gold['gold_passages']) == 0 or len(gold['answer']) == 0:
                    continue
                # Filter out query with no gold_associated_answer
                associated_answer_list = [g_p['associated_answers'] for g_p in gold['gold_passages'] if g_p['associated_answers']]
                if len(associated_answer_list) == 0:
                    continue

                gold_passages.append(gold)
            
            # For dev set, we random select 100 queries (the same as LOFT)
            if split == "dev":
                random.shuffle(gold_passages)
                gold_passages = gold_passages[:100]

            gold_passage_collection = {}
            for gold in gold_passages:
                gold_d = [d['text'] for d in gold['gold_passages']]
                gold_ans = list(set(gold['answer']))
                gold_passage_collection[gold['qid']] = (gold_d, gold_ans)


            # Now load the predictions from models
            filter_retrieval_predictions(models, dataset, split, gold_passage_collection, psg_corpus)
            
            
    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        default="nq,hotpotqa,fever,wow",
    )

    parser.add_argument(
        "--models",
        type=str,
        default="dpr_our,contriever_our,bm25_our,drqa,blink",
    )

    args = parser.parse_args()

    main(
            datasets=args.datasets,
            models=args.models,
        )


