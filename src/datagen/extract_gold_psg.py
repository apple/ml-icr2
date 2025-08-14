#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

import multiprocessing
from multiprocessing.pool import ThreadPool
import argparse
import json
import os
import spacy
from tqdm import tqdm

import kilt.kilt_utils as utils
from collections import defaultdict

print("loading knowledge source from {}".format("./corpus/kilt_document_corpus.json"), flush=True)
with open("./corpus/kilt_document_corpus.json", "r") as f:
    ks = json.load(f)



def create_chunk(document, buffer, paragraph_id, paragraph, section):
    start = buffer[0].idx
    end = buffer[-1].idx + len(buffer[-1])

    return {
        "_id": document["_id"],
        "wikipedia_id": document["wikipedia_id"],
        "wikipedia_title": document["wikipedia_title"],
        "text": paragraph.text[start : end + 1].strip(),
        "tmp_len": len(buffer),
        "section": section,
        "paragraph_id": paragraph_id
    }

def process_document(document, nlp, gold_paragraph_ids, gold_character_ids, chunk_size=100):
    # initialization
    buffer = []
    section = "Section::::Abstract"
    output = []

    # loop paragrpahs removing first (title)
    for paragraph_id, paragraph in enumerate(nlp.pipe(document["text"][1:]), 1):

        # if section then save name and move on
        if "Section::::" in paragraph.text:
            section = paragraph.text.strip()
            continue

        for sentence in paragraph.sents:
            if buffer and len(buffer) + len(sentence) >= chunk_size: # if buffer with new setnence satisfy chunk size
                # create new chunk
                new_chunk = create_chunk(
                    document, buffer, paragraph_id, paragraph, section
                )
                if paragraph_id in gold_paragraph_ids:
                    new_chunk['gold_passages'] = 1
                else:
                    new_chunk['gold_passages'] = 0
                output.append(new_chunk)
                buffer = []

            for token in sentence:
                word = token.text.strip()
                if word and len(word) > 0:
                    buffer.append(token)

        if buffer:
            # create new chunk
            new_chunk = create_chunk(
                document, buffer, paragraph_id, paragraph, section
            )
            if paragraph_id in gold_paragraph_ids:
                new_chunk['gold_passages'] = 1
            else:
                new_chunk['gold_passages'] = 0

            # conditions on merging with previous chunk
            if (
                output
                and document["wikipedia_id"] == output[-1]["wikipedia_id"]
                and section == output[-1]["section"]
                and len(buffer) + output[-1]["tmp_len"] < chunk_size
            ):
                # appending new data
                output[-1]["text"] += " " + new_chunk["text"]
                output[-1]["tmp_len"] += new_chunk["tmp_len"] + 1
                output[-1]["gold_passages"] += new_chunk["gold_passages"]
            else:
                output.append(new_chunk)
            buffer = []
    
    gold_output = []
    

    # print("Currently processing document:", gold_character_ids)
    if gold_character_ids:
        for gold_character_ids_anno in gold_character_ids:
            pids, csid, ceid = gold_character_ids_anno[0], gold_character_ids_anno[1], gold_character_ids_anno[2] - 1
            # print("Start processing annotation:", pids, csid, ceid)

            # We track each gold paragraph's processed length
            each_para_length_record = {}
            for pid in gold_paragraph_ids:
                each_para_length_record[pid] = 0
            
            for out in output:
                
                # Remove paragraph with character ids 
                # not in gold anwer's start and end character id
                if out['paragraph_id'] in pids:
                    paragraph_length = len(out['text'])
                    text_range = [i for i in range(each_para_length_record[pid], each_para_length_record[pid]+paragraph_length)]
                    
                    if csid in text_range or ceid in text_range:
                        out['gold_passages'] = 1
                        # print("Adding:")
                        # print(out['text'])
                        # print(text_range)
                        # print(csid, ceid)
                    else:
                        out['gold_passages'] = 0
                    each_para_length_record[pid] += len(out['text'])

                if "tmp_len" in out:
                    del out["tmp_len"]
                if out['gold_passages'] > 0:
                    gold_output.append(out)
    else:
        for out in output:
            del out["tmp_len"]
            if out['gold_passages'] > 0:
                gold_output.append(out)

    return gold_output

def run_thread(args):
    df = args["df"]
    nlp = args["nlp"]
    id = args["id"]
    chunk_size = args["chunk_size"]
    ks = args["ks"]

    if id == 0:
        iter_ = tqdm(df)
    else:
        iter_ = df

    
    outputs = []
    # For each query
    for annotation in iter_:
        output_dict = {}
        output_dict['qid'] = annotation['id']
        output_dict['input'] = annotation['input']
        output_dict['answer'] = []
        output_dict['gold_passages'] = []

        paragraph2answer = defaultdict(list)
        non_annotated_answer = []
        annotated_answer = []

        # For each gold annotation
        memory = {}
        for output in annotation['output']:
            
            # if this annotation has provenance, we need to add all provenance passage into gold, to avoid that adding false negative into context
            if "provenance" in list(output.keys()):

                # get mapping from docid to relevant paragraphs
                docid2gold_pid = defaultdict(list)
                docid2gold_cid = defaultdict(list)
                for provenance in output['provenance']:
                    start_paragraph_id, end_paragraph_id = provenance['start_paragraph_id'], provenance['end_paragraph_id']
                    docid2gold_pid[provenance['wikipedia_id']] += [i for i in range(start_paragraph_id, end_paragraph_id+1)]

                    if "start_character" in provenance.keys() and "end_character" in provenance.keys():
                        start_char_id, end_char_id = provenance['start_character'], provenance['end_character']    
                        docid2gold_cid[provenance['wikipedia_id']].append(([i for i in range(start_paragraph_id, end_paragraph_id+1)], start_char_id, end_char_id))
                    else:
                        docid2gold_cid = {}
                
                # process all relevant document and get gold passages
                for doc_id in docid2gold_pid:
                    try:
                        document = ks[doc_id]
                    except:
                        continue
                    gold_paragraph_ids = docid2gold_pid[doc_id]

                    try:
                        gold_char_ids = docid2gold_cid[doc_id]
                    except:
                        gold_char_ids = None
                    
                    # print("Process doc:", doc_id)
                    gold_passages = process_document(document=document, nlp=nlp, gold_paragraph_ids=gold_paragraph_ids, gold_character_ids=gold_char_ids, chunk_size=chunk_size)
                        
                    
                    output_dict['gold_passages'] += gold_passages

                    # find the mapping from gold passage to answer
                    for p_d in gold_passages:
                        # print("p_d, gold passage:", p_d)
                        if "answer" in list(output.keys()):
                            paragraph2answer[p_d['text']].append(output['answer']) 
                            annotated_answer.append(output['answer'])

                
            elif "answer" in list(output.keys()) and "provenance" not in list(output.keys()):
                non_annotated_answer.append(output['answer'])
            else:
                continue
        
        # print(output_dict)

        # Adding associated answer to each paragraph
        # removing duplicate gold passages
        deduplicate_output_dict = []
        deduplicate_ = []
        # print(output_dict['gold_passages'])
        for p in output_dict['gold_passages']:
            if p['text'] in deduplicate_:
                continue
            else:
                if p['text'] in paragraph2answer.keys():
                    p['associated_answers'] = paragraph2answer[p['text']]
                else:
                    p['associated_answers'] = None
                del p['gold_passages']
                del p['paragraph_id']
                deduplicate_.append(p['text'])
                deduplicate_output_dict.append(p)

        output_dict['gold_passages'] = deduplicate_output_dict

        output_dict['answer'] = annotated_answer + non_annotated_answer

        # print(output_dict)

        outputs.append(output_dict)

    return outputs

def load_dataset(dataset):
    with open(dataset, "r") as f:
        df = [json.loads(d.strip()) for d in f.readlines()]
    return df


def main(dataset, num_threads, folder, chunk_size):

    print("loading dataset {}".format(dataset), flush=True)
    df = load_dataset(dataset)
    
    arguments = [
        {
            "id": id,
            "df": chunk,
            "ks": ks,
            "nlp": spacy.load("en_core_web_sm"),
            "chunk_size": chunk_size,
        }
        for id, chunk in enumerate(utils.chunk_it(df, num_threads))
    ]

    print("starting {} threads..".format(num_threads))
    pool = ThreadPool(num_threads)
    results = pool.map(run_thread, arguments)

    filename = dataset.split("/")[-1].replace("kilt", "gold-passage")

    merge_outputs = []
    for thread_results in results:
        for outline in thread_results:
            merge_outputs.append(json.dumps(outline)+"\n")
    
    with open(os.path.join("/mnt/task_runtime/gold_passages_debug/", filename), "w+", encoding="utf-8") as f:
        f.writelines(merge_outputs)

    pool.terminate()
    pool.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="our_data/hotpotqa-dev-kilt.jsonl",
    )

    parser.add_argument(
        "--chunk_size", default=100, type=int, help="chunk max token size",
    )

    parser.add_argument(
        "--folder", type=str, help="path where to save and load files", default="/mnt/task_runtime/gold_passages_debug/"
    )

    parser.add_argument(
        "--threads", type=int, help="number of threads", default=1
    )

    args = parser.parse_args()

    if args.threads == None:
        args.threads = int(multiprocessing.cpu_count())

    main(
            dataset=args.dataset,
            num_threads=args.threads,
            folder=args.folder,
            chunk_size=args.chunk_size,
        )


