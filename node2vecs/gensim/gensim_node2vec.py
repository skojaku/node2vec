# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-18 07:15:16
import numpy as np
from ..node2vec import Node2Vec
import gensim
from tqdm import tqdm


class GensimNode2Vec(Node2Vec):
    def __init__(self, **params):
        super().__init__(**params)
        self.w2vparams = {
            "sg": 1,
            "min_count": 0,
            "epochs": self.epochs,
            "workers": 4,
            "negative": self.negative,
            "alpha": self.alpha,
            "ns_exponent": self.ns_exponent,
        }

    #        if params is not None:
    #            for k, v in params.items():
    #                self.w2vparams[k] = v

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        self.w2vparams["window"] = self.window
        self.w2vparams["vector_size"] = dim

        def pbar(it):
            return tqdm(it, desc="Training", total=self.num_walks * self.num_nodes)

        starting_nodes = np.kron(
            np.arange(self.num_nodes),
            np.ones(self.num_walks),
        ).astype(np.int64)
        np.random.shuffle(starting_nodes)

        seqs = self.sampler.sampling(starting_nodes).tolist()
        self.model = gensim.models.Word2Vec(sentences=pbar(seqs), **self.w2vparams)

        num_nodes = len(self.model.wv.index_to_key)
        self.in_vec = np.zeros((num_nodes, dim))
        self.out_vec = np.zeros((num_nodes, dim))
        for i in range(num_nodes):
            if i not in self.model.wv:
                continue
            self.in_vec[i, :] = self.model.wv[i]
            self.out_vec[i, :] = self.model.syn1neg[self.model.wv.key_to_index[i]]
