# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-24 22:56:34
import random

import numpy as np
from numba import njit
from torch.utils.data import Dataset

from node2vecs.utils.random_walks import RandomWalkSampler


class TripletDataset(Dataset):
    """Dataset for training word2vec with negative sampling."""

    def __init__(
        self,
        adjmat,
        num_walks,
        window_length,
        noise_sampler,
        padding_id,
        walk_length=40,
        epochs=1,
        p=1.0,
        q=1.0,
        context_window_type="double",
        buffer_size=100000,
        negative=5,
    ):
        """Dataset for training word2vec with negative sampling.
        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: scipy sparse matrix format (csr).
        :param num_walks: Number of random walkers per node
        :type num_walks: int
        :param window_length: length of the context window
        :type window_length: int
        :param noise_sampler: Noise sampler
        :type noise_sampler: NodeSampler
        :param padding_id: Index of the padding node
        :type padding_id: int
        :param walk_length: length per walk, defaults to 40
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        :param context_window_type: The type of context window. `context_window_type="double"` specifies a context window that extends both left and right of a focal node. context_window_type="left" and ="right" specifies that extends left and right, respectively.
        :type context_window_type: str, optional
        :param buffer_size: Buffer size for sampled center and context pairs, defaults to 10000
        :type buffer_size: int, optional
        """
        self.negative = negative
        self.adjmat = adjmat
        self.num_walks = num_walks
        self.window_length = window_length
        self.noise_sampler = noise_sampler
        self.walk_length = walk_length
        self.padding_id = padding_id
        self.epochs = epochs
        self.context_window_type = {"double": 0, "left": -1, "right": 1}[
            context_window_type
        ]
        self.rw_sampler = RandomWalkSampler(
            adjmat, walk_length=walk_length, p=p, q=q, padding_id=padding_id
        )
        self.node_order = np.random.choice(
            adjmat.shape[0], adjmat.shape[0], replace=False
        )
        self.n_nodes = adjmat.shape[0]

        self.ave_deg = adjmat.sum() / adjmat.shape[0]

        # Counter and Memory
        self.n_sampled = 0
        self.sample_id = 0
        self.scanned_node_id = 0
        self.buffer_size = buffer_size
        self.contexts = None
        self.centers = None
        self.random_contexts = None

        # Initialize
        self._generate_samples()

    def __len__(self):
        return self.epochs * self.n_nodes * self.num_walks * self.walk_length

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self._generate_samples()

        center = self.centers[self.sample_id].astype(np.int64)
        cont = self.contexts[self.sample_id, :].astype(np.int64)
        rand_cont = self.random_contexts[self.sample_id, :].astype(np.int64)

        self.sample_id += 1

        return center, cont, rand_cont

    def _generate_samples(self):
        next_scanned_node_id = np.minimum(
            self.scanned_node_id + self.buffer_size, self.n_nodes
        )
        walks = self.rw_sampler.sampling(
            self.node_order[self.scanned_node_id : next_scanned_node_id]
        )
        self.centers, self.contexts = _get_center_context(
            context_window_type=self.context_window_type,
            walks=walks,
            n_walks=walks.shape[0],
            walk_len=walks.shape[1],
            window_length=self.window_length,
            padding_id=self.padding_id,
        )
        s = self.centers != self.padding_id

        self.centers, self.contexts = self.centers[s], self.contexts[s, :]
        self.random_contexts = np.hstack(
            [
                self.noise_sampler.sampling(
                    center_nodes=self.centers,
                    context_nodes=self.contexts[:, i],
                    # padding_id=self.padding_id,
                ).reshape((-1, 1))
                for i in range(self.negative * self.contexts.shape[1])
            ]
        )

        self.n_sampled = len(self.centers)
        self.scanned_node_id = next_scanned_node_id % self.n_nodes
        self.sample_id = 0

        # Shuffle
        # ids = np.arange(self.n_sampled, dtype=int)
        # np.random.shuffle(ids)
        # self.centers, self.contexts, self.random_contexts = (
        #    self.centers[ids],
        #    self.contexts[ids],
        #    self.random_contexts[ids],
        # )


class ModularityDataset(TripletDataset):
    def __init__(self, **params):
        super().__init__(**params)

    def __getitem__(self, idx):
        if self.sample_id == self.n_sampled:
            self._generate_samples()

        center = self.centers[self.sample_id].astype(np.int64)
        cont = self.contexts[self.sample_id, :].astype(np.int64)
        rand_cont = self.random_contexts[self.sample_id, :].astype(np.int64)
        base_center = np.random.randint(0, self.n_nodes, size=center.shape)
        base_cont = np.random.randint(0, self.n_nodes, size=cont.shape)

        self.sample_id += 1

        return center, cont, rand_cont, base_center, base_cont


def _get_center_context(
    context_window_type, walks, n_walks, walk_len, window_length, padding_id
):
    """Get center and context pairs from a sequence
    window_type = {-1,0,1} specifies the type of context window.
    window_type = 0 specifies a context window of length window_length that extends both
    left and right of a center word. window_type = -1 and 1 specifies a context window
    that extends either left or right of a center word, respectively.
    """
    if context_window_type == 0:
        center, context = _get_center_double_context_windows(
            walks, n_walks, walk_len, window_length, padding_id
        )
    elif context_window_type == -1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
        )
    elif context_window_type == 1:
        center, context = _get_center_single_context_window(
            walks, n_walks, walk_len, window_length, padding_id, is_left_window=False
        )
    else:
        raise ValueError("Unknown window type")
    # center = np.outer(center, np.ones(context.shape[1]))
    # center, context = center.reshape(-1), context.reshape(-1)
    # s = (center != padding_id) * (context != padding_id)
    # center, context = center[s], context[s]
    # order = np.arange(len(center))
    # random.shuffle(order)
    # return center[order].astype(int), context[order].astype(int)
    return center.astype(int), context.astype(int)


@njit(nogil=True)
def _get_center_double_context_windows(
    walks, n_walks, walk_len, window_length, padding_id
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones(
        (n_walks * walk_len, 2 * window_length), dtype=np.int64
    )
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        for i in range(window_length):
            if t_walk - 1 - i < 0:
                break
            contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]

        for i in range(window_length):
            if t_walk + 1 + i >= walk_len:
                break
            contexts[start:end, window_length + i] = walks[:, t_walk + 1 + i]

    return centers, contexts


@njit(nogil=True)
def _get_center_single_context_window(
    walks, n_walks, walk_len, window_length, padding_id, is_left_window=True
):
    centers = padding_id * np.ones(n_walks * walk_len, dtype=np.int64)
    contexts = padding_id * np.ones((n_walks * walk_len, window_length), dtype=np.int64)
    for t_walk in range(walk_len):
        start, end = n_walks * t_walk, n_walks * (t_walk + 1)
        centers[start:end] = walks[:, t_walk]

        if is_left_window:
            for i in range(window_length):
                if t_walk - 1 - i < 0:
                    break
                contexts[start:end, window_length - 1 - i] = walks[:, t_walk - 1 - i]
        else:
            for i in range(window_length):
                if t_walk + 1 + i >= walk_len:
                    break
                contexts[start:end, i] = walks[:, t_walk + 1 + i]
    return centers, contexts
