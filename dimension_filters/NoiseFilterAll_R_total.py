import numpy as np
import pandas as pd
from .AbstractFilter import AbstractFilter
import random

class NoiseFilterAll_RT(AbstractFilter):
    """
    """
    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]
        self.hyperparams = kwargs["hyperparams"]

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)
        dlist = self.run[(self.run.query_id == query.query_id)].doc_id.to_list()
        all_docs = self.docs_encoder.mapping.index.copy()

        ### POSITIVE PART
        npos = self.hyperparams['npos'] ##2
        pos_demb = np.mean(self.docs_encoder.get_encoding(dlist[:npos]), axis=0)  
        pos_score = self.hyperparams['a']*np.multiply(qemb, pos_demb)
        ## NEGATIVE PART
        nneg = self.hyperparams['nneg'] ##3
        all_docs = all_docs.difference(dlist[:npos])  # remove i relevant
        neg_dlist = all_docs.to_series().sample(n=nneg).to_list()   # sample k neg
        neg_demb = np.mean(self.docs_encoder.get_encoding(neg_dlist), axis=0) 
        neg_score = self.hyperparams['b']*np.multiply(qemb, neg_demb)
        #a, b = 0.9, 0.3
        itx_vec =  pos_score - neg_score

        return itx_vec