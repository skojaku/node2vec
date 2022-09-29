import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

#
# Embedding model
#
class Word2Vec(nn.Module):
    def __init__(self, n_nodes, dim, normalize=False):
        super(Word2Vec, self).__init__()
        # Layers
        self.ivectors = torch.nn.Embedding(n_nodes, dim, dtype=torch.float)
        self.ovectors = torch.nn.Embedding(n_nodes, dim, dtype=torch.float)
        self.n_nodes = n_nodes
        # Parameters
        self.ovectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, dim),
                    FloatTensor(n_nodes, dim).uniform_(-0.5 / dim, 0.5 / dim),
                ]
            )
        )
        self.ivectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, dim),
                    FloatTensor(n_nodes, dim).uniform_(-0.5 / dim, 0.5 / dim),
                ]
            )
        )

    def forward(self, data):
        x = self.ivectors(data)
        if self.training is False:
            if self.ivectors.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x

    def forward_i(self, data):
        x = self.ivectors(data)
        if self.training is False:
            if self.ivectors.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x

    def forward_o(self, data):
        x = self.ovectors(data)
        if self.training is False:
            if self.ovectors.weight.is_cuda:
                return x.detach().cpu().numpy()
            else:
                return x.detach().numpy()
        else:
            return x

    def embedding(self, data=None, return_out_vector=False):
        """Generate an embedding. If data is None, generate an embedding of all noddes"""
        if data is None:
            data = torch.arange(self.n_nodes)
        if return_out_vector:
            emb = self.forward_i(data)
        else:
            emb = self.forward_o(data)
        return emb
