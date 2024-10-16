import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
sys.path.append(".")

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
import dimension_filters
from memmap_interface import MemmapCorpusEncoding, MemmapQueriesEncoding
from sentence_transformers import SentenceTransformer

import ir_measures
def compute_measure(run, qrels, measure_name):
    measure = [ir_measures.parse_measure(measure_name)]
    out = pd.DataFrame(ir_measures.iter_calc(measure, qrels, run))
    out['measure'] = out['measure'].astype(str)
    return out

os.environ["TOKENIZERS_PARALLELISM"] = "false"

collection2corpus = {"deeplearning19": "msmarco-passage", 
                     "deeplearning20": "msmarco-passage",
                     "deeplearninghd": "msmarco-passage", 
                     "robust04": "robust04"}

m2hf = {"tasb": 'sentence-transformers/msmarco-distilbert-base-tas-b',
        "contriever": "facebook/contriever-msmarco",
        "ance": "sentence-transformers/msmarco-roberta-base-ance-firstp"}

def main(args, index, hyperparams, comb):

    # read the queries
    query_reader_params = {'sep': "\t", 'names': ["query_id", "text"], 'header': None, 'dtype': {"query_id": str}}
    queries = pd.read_csv(f"{args.datadir}/queries/{args.collection}/queries.tsv", **query_reader_params)

    # read qrels
    qrels_reader_params = {'sep': " ", 'names': ["query_id", "doc_id", "relevance"], "usecols": [0,2,3],
                            'header': None, "dtype": {"query_id": str, "doc_id": str}}
    qrels = pd.read_csv(f"{args.datadir}/qrels/{args.collection}/qrels.txt", **qrels_reader_params)

    print('Number of og queries: ', len(queries.query_id.unique()))
    print('Number of og queries in qrels: ', len(qrels.query_id.unique()))

    # keep only queries with relevant docs
    queries = queries.loc[queries.query_id.isin(qrels.query_id)]
    print('Final n.queries: ', len(queries))

    # load memmap for the corpus
    corpora_memmapsdir = f"{args.datadir}/memmaps/{args.model_name}/corpora/{collection2corpus[args.collection]}"
    docs_encoder = MemmapCorpusEncoding(f"{corpora_memmapsdir}/corpus.dat",
                                        f"{corpora_memmapsdir}/corpus_mapping.csv")

    memmapsdir = f"{args.datadir}/memmaps/{args.model_name}/{args.collection}"
    qrys_encoder = MemmapQueriesEncoding(f"{memmapsdir}/queries.dat",
                                        f"{memmapsdir}/queries_mapping.tsv")
    

    run_reader_parmas = {'names': ["query_id", "doc_id", "rank", "score"], 'usecols': [0, 2, 3, 4], 'sep': "\t",
                    'dtype': {"query_id": str, "doc_id": str, "rank": int, "score": float}}
    run = pd.read_csv(f"{args.datadir}/runs/{args.collection}/{args.model_name}.tsv", **run_reader_parmas)
    model = SentenceTransformer(m2hf[args.model_name])
    answers_path = f"{args.datadir}/runs/{args.collection}/chatgpt4_answer.csv"
    filtering = dimension_filters.NoiseFilterGPT(qrels=qrels, docs_encoder=docs_encoder, qrys_encoder=qrys_encoder,
                                                model=model, answers_path=answers_path, run=run, hyperparams=hyperparams)

    # mapper
    mapping = pd.read_csv(f"{corpora_memmapsdir}/corpus_mapping.csv", dtype={'did': str})
    mapper = mapping.set_index('offset').did.to_list()

    rel_dims = filtering.filter_dims(queries, explode=True)
    qembs = qrys_encoder.get_encoding(queries.query_id.to_list()) 
    q2r = pd.DataFrame({"query_id": queries.query_id.to_list(), "row": np.arange(len(queries.query_id.to_list()))})

    def masked_retrieve_and_evaluate(qembs, dim_importance, alpha, measure):
        n_dims = int(np.round(alpha * qrys_encoder.shape[1]))
        selected_dims = dim_importance.loc[dim_importance["drank"] <= n_dims][["query_id", "dim"]]

        rows = np.array(selected_dims[["query_id"]].merge(q2r)["row"])
        cols = np.array(selected_dims["dim"])

        mask = np.zeros_like(qembs)
        mask[rows, cols] = 1
        enc_queries = np.where(mask, qembs, 0)

        ip, idx = index.search(enc_queries, 1000)
        nqueries = len(ip)

        out = []
        for i in range(nqueries):
            local_run = pd.DataFrame({"query_id": queries.iloc[i]["query_id"], "doc_id": idx[i], "score": ip[i]})
            local_run.sort_values("score", ascending=False, inplace=True)
            local_run['doc_id'] = local_run['doc_id'].apply(lambda x: mapper[x])
            out.append(local_run)

        out = pd.concat(out)

        res = compute_measure(out, qrels, measure)
        res["alpha"] = alpha

        return res

    filter_name = hyperparams['filtername']
    measure_name = hyperparams['measure']
    alphas = np.round(np.arange(0.1, 1.1, 0.1), 2)

    result = []
    for alpha in alphas:
        output = masked_retrieve_and_evaluate(qembs, rel_dims, alpha, measure_name)
        result.append(output)
        print(f'{measure_name}@{alpha}: ', output.value.mean())

    result = pd.concat(result)
    result.to_csv(f"{args.datadir}/performance/performance_gpt_2/{args.collection}_{args.model_name}_{filter_name}_{measure_name}_{comb}.csv", index=False)


if __name__ == "__main__":
    
    tqdm.pandas()

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--collection", default="deeplearning20")
    parser.add_argument("-r", "--model_name", default="contriever")
    parser.add_argument("-d", "--datadir", default="data")

    args = parser.parse_args()

    import pickle
    #if args.collection != 'robust04':
    hyperparams_filename = f"data/performance/performance_gpt_2/hyperparams.pickle"
    #else:
    #    hyperparams_filename = f"data/performance/performance_gpt/hyperparams_ROBUST04.pickle" 
    
    with open(hyperparams_filename, 'rb') as handle:
        combinations_dicts = pickle.load(handle)
    print('processing n_conf', len(combinations_dicts))

    print('load_index')
    # load faiss index
    faiss_path = f"{args.datadir}/vectordb/{args.model_name}/corpora/{collection2corpus[args.collection]}/"
    index_name = "index_db.faiss"
    index = faiss.read_index(faiss_path + index_name)
    
    for i, hyperparams in tqdm(enumerate(combinations_dicts), total=len(combinations_dicts)):
        print('hyperparms: ', hyperparams)
        main(args, index, hyperparams, i)