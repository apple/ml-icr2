#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#

from __future__ import division

import json
import tqdm as tqdm
import argparse
from kilt import kilt_utils as utils



import math

def chunked(iterable, n):
    """ Split iterable into ``n`` iterables of similar size

    Examples::
        >>> l = [1, 2, 3, 4]
        >>> list(chunked(l, 4))
        [[1], [2], [3], [4]]

        >>> l = [1, 2, 3]
        >>> list(chunked(l, 4))
        [[1], [2], [3], []]

        >>> l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(chunked(l, 4))
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10]]

    """
    chunksize = int(math.ceil(len(iterable) / n))
    return (iterable[i * chunksize:i * chunksize + chunksize]
            for i in range(n))

def main(args):

    # load configs
    with open(args.test_config, "r") as fin:
        test_config_json = json.load(fin)

        print(test_config_json)

    for task_family, datasets in test_config_json.items():
        print("TASK: {}".format(task_family))
        for dataset_name, dataset_file in datasets.items():
            print("Loading from", dataset_file)

            dataset_prefix = ''.join(dataset_file.split(".")[:-1])
            dataset_postfix = dataset_file.split(".")[-1]

            config_prefix = ''.join(args.test_config.split(".")[:-1])
            config_postfix = args.test_config.split(".")[-1]

            raw_data = utils.load_data(dataset_file)
            
            chunk_data = chunked(raw_data, args.thread)

            for i, chunk in enumerate(chunk_data):
                chunk_dataset_path = dataset_prefix+"-"+str(i)+"."+dataset_postfix
                with open(chunk_dataset_path, "w+") as f:
                    f.writelines([json.dumps(d) + '\n' for d in chunk])

                new_config = test_config_json
                new_config[task_family][dataset_name] = chunk_dataset_path
                with open(config_prefix+"-"+str(i)+"."+config_postfix, "w+") as f:
                    json.dump(new_config, f)

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_config",
        dest="test_config",
        type=str,
        default="kilt/configs/our_test_data.json",
        help="Test Configuration.",
    )

    parser.add_argument(
        "--thread",
        type=int,
        default=16,
        help="How many compute nodes to execute retrieval.",
    )

    args = parser.parse_args()

    main(args)

