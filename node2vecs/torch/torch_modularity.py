# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:38:28
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-19 07:09:39
import numpy as np
from . import utils
from node2vecs.node2vec import Node2Vec
from .models import Word2Vec
from .loss import ModularityTripletLoss
from .dataset import ModularityDataset
from .train import train
from node2vecs.utils.node_sampler import ConfigModelNodeSampler
import torch
from tqdm import tqdm
from torch.optim import AdamW, Adam, SGD, SparseAdam
from torch.utils.data import DataLoader


class TorchModularity(Node2Vec):
    def __init__(
        self,
        batch_size=256,
        device="cpu",
        buffer_size=100000,
        context_window_type="double",
        miniters=200,
        num_workers=1,
        alpha=1e-3,
        **params
    ):
        """Residual2Vec based on the stochastic gradient descent.
        :param noise_sampler: Noise sampler
        :type noise_sampler: NodeSampler
        :param window_length: length of the context window, defaults to 10
        :type window_length: int
        :param batch_size: Number of batches for the SGD, defaults to 4
        :type batch_size: int
        :param num_walks: Number of random walkers per node, defaults to 100
        :type num_walks: int
        :param walk_length: length per walk, defaults to 80
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        :param buffer_size: Buffer size for sampled center and context pairs, defaults to 10000
        :type buffer_size: int, optional
        :param context_window_type: The type of context window. `context_window_type="double"` specifies a context window that extends both left and right of a focal node. context_window_type="left" and ="right" specifies that extends left and right, respectively.
        :type context_window_type: str, optional
        :param miniter: Minimum number of iterations, defaults to 200
        :type miniter: int, optional
        """
        super().__init__(**params)
        self.noise_sampler = ConfigModelNodeSampler(self.ns_exponent)
        self.device = device
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.miniters = miniters
        self.context_window_type = context_window_type
        self.num_workers = num_workers
        self.alpha = alpha

    def fit(self, adjmat):
        """Learn the graph structure to generate the node embeddings.
        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: numpy.ndarray or scipy sparse matrix format (csr).
        :return: self
        :rtype: self
        """

        # Convert to scipy.sparse.csr_matrix format
        adjmat = utils.to_adjacency_matrix(adjmat)

        # Set up the graph object for efficient sampling
        self.adjmat = adjmat
        self.n_nodes = adjmat.shape[0]
        self.noise_sampler.fit(adjmat)
        return self

    def update_embedding(self, dim):
        """Generate embedding vectors.
        :param dim: Dimension
        :type dim: int
        :return: Embedding vectors
        :rtype: numpy.ndarray of shape (num_nodes, dim), where num_nodes is the number of nodes.
          Each ith row in the array corresponds to the embedding of the ith node.
        """

        # Set up the embedding model
        PADDING_IDX = self.n_nodes
        model = Word2Vec(
            vocab_size=self.n_nodes + 1, embedding_size=dim, padding_idx=PADDING_IDX
        )
        model = model.to(self.device)
        loss_func = ModularityTripletLoss(n_neg=self.negative)

        # Set up the Training dataset
        adjusted_num_walks = np.ceil(
            self.num_walks
            * np.maximum(
                1,
                self.batch_size
                * self.miniters
                / (self.n_nodes * self.num_walks * self.rw_params["walk_length"]),
            )
        ).astype(int)
        self.rw_params["num_walks"] = adjusted_num_walks
        dataset = ModularityDataset(
            adjmat=self.adjmat,
            window_length=self.window,
            noise_sampler=self.noise_sampler,
            padding_id=PADDING_IDX,
            buffer_size=self.buffer_size,
            context_window_type=self.context_window_type,
            epochs=self.epochs,
            negative=self.negative,
            **self.rw_params
        )

        train(
            model=model,
            dataset=dataset,
            loss_func=loss_func,
            batch_size=self.batch_size,
            device=self.device,
            learning_rate=self.alpha,
            num_workers=4,
        )
        model.eval()
        self.in_vec = model.ivectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        self.out_vec = model.ovectors.weight.data.cpu().numpy()[:PADDING_IDX, :]
        return self.in_vec
