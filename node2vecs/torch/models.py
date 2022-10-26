# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-24 23:47:35
"""Word2Vec.
This module is a modified version of the Word2Vec module in
https://github.com/theeluwin/pytorch-sgn
"""
import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, learn_outvec=True):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learn_outvec = learn_outvec
        self.ivectors = nn.Embedding(
            self.vocab_size + 1,
            self.embedding_size,
            padding_idx=padding_idx,
            sparse=True,
        )
        self.ovectors = nn.Embedding(
            self.vocab_size + 1,
            self.embedding_size,
            padding_idx=padding_idx,
            sparse=True,
        )
        torch.nn.init.uniform_(
            self.ovectors.weight, -0.5 / embedding_size, 0.5 / embedding_size
        )
        torch.nn.init.uniform_(
            self.ivectors.weight, -0.5 / embedding_size, 0.5 / embedding_size
        )
        # nn.init.xavier_uniform_(self.ivectors.weight)
        # nn.init.xavier_uniform_(self.ovectors.weight)

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        # v = LongTensor(data)
        # v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(data)

    def forward_o(self, data):
        # v = LongTensor(data)
        # v = v.cuda() if self.ovectors.weight.is_cuda else v
        if self.learn_outvec:
            return self.ovectors(data)
        else:
            return self.ivectors(data)

    def embedding(self, return_out_vector=False):
        if return_out_vector is False:
            if self.ivectors.weight.is_cuda:
                return self.ivectors.weight.data.cpu().numpy()[: self.vocab_size]
            else:
                return self.ivectors.weight.data.numpy()[: self.vocab_size]
        else:
            if self.ovectors.weight.is_cuda:
                return self.ovectors.weight.data.cpu().numpy()[: self.vocab_size]
            else:
                return self.ovectors.weight.data.numpy()[: self.vocab_size]
