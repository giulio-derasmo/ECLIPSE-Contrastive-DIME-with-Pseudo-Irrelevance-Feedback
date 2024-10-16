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

coll2corpus = { "deeplearning19": "/hdd4/giuder/progetti/DIME/data/collections/collection.tsv",
                "robust04": "/hdd4/giuder/progetti/DIME/data/collections/robust04.csv",
                "corpus": "path2corpus"}

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--model_name")
    parser.add_argument("-p", "--data_dir", default="data")
    parser.add_argument("-c", "--collection", default="deeplearning19")
    args = parser.parse_args()

    corpus = "msmarco-passage" if args.collection in ["deeplearning19", "deeplearning20"] else "robust04"

    if corpus == "msmarco-passage":
        collection = load_collection(coll2corpus[args.collection])
        docs = pd.DataFrame([[id, normalize(text)] for id, text in collection.items()], columns=["did", "text"])
    else: 
        docs = pd.read_csv(coll2corpus[args.collection])
        docs['text'] = docs['text'].apply(lambda x: normalize(x)) 
 
    
    docs = docs.sort_values("did")
    docs["offset"] = np.arange(len(docs.index))
    model = SentenceTransformer(m2hf[args.model_name])

    fp = np.memmap(f"{args.data_dir}/memmaps/{args.model_name}/corpora/{corpus}/corpus.dat", dtype='float32', mode='w+', shape=(len(docs), 768))
    step = 6000
    for i in tqdm(range(0, len(docs.text.to_list()), step)):
        # Compute the embeddings 
        fp[i:i + step] = model.encode(docs.text.to_list()[i:i + step])

    docs[["did", "offset"]].to_csv(f"{args.data_dir}/memmaps/{args.model_name}/corpora/{corpus}/corpus_mapping.csv", index=False)