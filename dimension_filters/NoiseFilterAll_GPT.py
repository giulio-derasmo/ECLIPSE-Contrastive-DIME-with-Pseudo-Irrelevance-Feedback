import numpy as np
import pandas as pd
from .AbstractFilter import AbstractFilter
import random

class NoiseFilterGPT(AbstractFilter):
    """
    """
    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.hyperparams = kwargs["hyperparams"]

        self.model = kwargs["model"]
        self.gpt_answers = pd.read_csv(kwargs["answers_path"], dtype={"query_id": str})
        self.gpt_answers = self.gpt_answers.loc[~self.gpt_answers["query_id"].duplicated()].set_index("query_id")

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)
        dlist = self.run[(self.run.query_id == query.query_id)].doc_id.to_list()
        #dlist = self.run[(self.run.query_id == query.query_id)].iloc[:self.hyperparams['cutoff']].doc_id.to_list()

        if self.hyperparams['noise_filter'] == "1":
            ### POSITIVE PART
            pos_demb = self.model.encode(self.gpt_answers.loc[query.query_id, "text"]) 
            pos_score = self.hyperparams['a']*np.multiply(qemb, pos_demb)
            ## NEGATIVE PART
            nneg = self.hyperparams['nneg'] ##3
            neg_demb = np.mean(self.docs_encoder.get_encoding(dlist[-nneg:]), axis=0) 
            neg_score = self.hyperparams['b']*np.multiply(qemb, neg_demb)
            itx_vec =  pos_score - neg_score

        return itx_vec