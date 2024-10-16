import os 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import sys
sys.path.append('.') 

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.autonotebook import tqdm
from time import time
import argparse
from normalize_text import normalize
import pandas as pd
from utils import load_collection


m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "glove": 'sentence-transformers/average_word_embeddings_glove.6B.300d',
        "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--model_name")
    parser.add_argument("-p", "--data_dir", default="data")
    parser.add_argument("-c", "--collection", default="deeplearning19")
    args = parser.parse_args()


    queries = []
    qpath = f"{args.data_dir}/queries/{args.collection}/queries.tsv"
    queries = pd.read_csv(qpath, sep="\t", header=None, names=["qid", "text"], dtype={"qid": str})
    queries["offset"] = np.arange(len(queries.index))
    model = SentenceTransformer(m2hf[args.model_name])

    repr = np.array(queries.text.apply(model.encode).to_list())
    fp = np.memmap(f"{args.data_dir}/memmaps/{args.model_name}/{args.collection}/queries.dat", dtype='float32', mode='w+', shape=repr.shape)
    fp[:] = repr[:]
    fp.flush()
    
    queries[['qid', 'offset']].to_csv(f"{args.data_dir}/memmaps/{args.model_name}/{args.collection}/queries_mapping.tsv", sep="\t", index=False)
    