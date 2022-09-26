import numpy as np
from scipy import sparse
from numba import njit
from collections.abc import Iterable
from torch.utils.data import Dataset



#
# Walk sampler
#
class RandomWalkSampler:
    """Module for generating a sentence using random walks.
    .. highlight:: python
    .. code-block:: python
        >>> from residual2vec.random_walk_sampler import RandomWalkSampler
        >>> net = nx.adjacency_matrix(nx.karate_club_graph())
        >>> sampler = RandomWalkSampler(net)
        >>> walk = sampler.sampling(start=12)
        >>> print(walk) # [12, 11, 10, 9, ...]
    """

    def __init__(self, adjmat, walk_length=40, num_walks = 10, p=1, q=1):
        """Random Walk Sampler.
        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: scipy sparse matrix format (csr).
        :param num_walks: number of walkers per node, defaults to 10
        :type num_walks: int, optional
        :param walk_length: length per walk, defaults to 40
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        """
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weighted = (~np.isclose(np.min(adjmat.data), 1)) or (
            ~np.isclose(np.max(adjmat.data), 1)
        )
        self.num_nodes = adjmat.shape[0]

        adjmat.sort_indices()
        self.indptr = adjmat.indptr.astype(np.int64)
        self.indices = adjmat.indices.astype(np.int64)
        if self.weighted:
            data = adjmat.data / adjmat.sum(axis=1).A1.repeat(np.diff(self.indptr))
            self.data = _csr_row_cumsum(self.indptr, data)

    def __iter__(self):
        for i in range(self.num_walks):
            for node_id in np.random.choice(self.num_nodes, size=self.num_nodes, replace=False):
                yield self._sampling(node_id).tolist()

    def __len__(self):
        return self.num_walks * self.num_walks

    def sampling(self, start = None, num_walks = None):

        if num_walks is None:
            num_walks = self.num_walks

        if start is None:
            start = np.arange(self.n_nodes, dtype=np.int64)

        if isinstance(start, Iterable):
           # iterable
           walks = [self._sampling(s) for _ in range(num_walks) for s in start]
        else:
           walks = [self._sampling(start) for _ in range(num_walks)]
        return walks

    def _sampling(self, start):
        """Sample a random walk path.
        :param start: ID of the starting node
        :type start: int
        :return: array of visiting nodes
        :rtype: np.ndarray
        """
        padding_id = -1
        if self.weighted:
            walk = _random_walk_weighted(
                self.indptr,
                self.indices,
                self.data,
                self.walk_length,
                self.p,
                self.q,
                padding_id=padding_id,
                ts=start
                if isinstance(start, Iterable)
                else np.array([start]).astype(np.int64),
            )
        else:
            walk = _random_walk(
                self.indptr,
                self.indices,
                self.walk_length,
                self.p,
                self.q,
                padding_id=padding_id,
                ts=start
                if isinstance(start, Iterable)
                else np.array([start]).astype(np.int64),
            )
        walk = walk.astype(np.int64)
        walk = get_walk_seq(walk, walk.shape[0], walk.shape[1], padding_id=padding_id)
        if isinstance(start, Iterable):
            return walk
        else:
            return walk[0]


@njit(nogil=True)
def get_walk_seq(walks, n_walks, n_steps, padding_id):
    retval = []
    for i in range(n_walks):
        w = walks[i, :]
        retval.append(w[w != padding_id])
    return retval


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _isin_sorted(a, x):
    return a[np.searchsorted(a, x)] == x


@njit(nogil=True)
def _random_walk(indptr, indices, walk_length, p, q, padding_id, ts):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = padding_id * np.ones((len(ts), walk_length), dtype=indices.dtype)
    for walk_id, t in enumerate(ts):
        walk[walk_id, 0] = t
        neighbors = _neighbors(indptr, indices, t)
        if len(neighbors) == 0:
            continue
        walk[walk_id, 1] = np.random.choice(neighbors)
        for j in range(2, walk_length):
            neighbors = _neighbors(indptr, indices, walk[walk_id, j - 1])
            if len(neighbors) == 0:
                break

            if p == q == 1:
                # faster version
                walk[walk_id, j] = np.random.choice(neighbors)
                continue
            while True:
                new_node = np.random.choice(neighbors)
                r = np.random.rand()
                if new_node == walk[walk_id, j - 2]:
                    if r < prob_0:
                        break
                elif _isin_sorted(
                    _neighbors(indptr, indices, walk[walk_id, j - 2]), new_node
                ):
                    if r < prob_1:
                        break
                elif r < prob_2:
                    break
            walk[walk_id, j] = new_node
    return walk


@njit(nogil=True)
def _random_walk_weighted(indptr, indices, data, walk_length, p, q, padding_id, ts):
    max_prob = max(1 / p, 1, 1 / q)
    prob_0 = 1 / p / max_prob
    prob_1 = 1 / max_prob
    prob_2 = 1 / q / max_prob

    walk = padding_id * np.ones((len(ts), walk_length), dtype=indices.dtype)

    for walk_id, t in enumerate(ts):
        walk[walk_id, 0] = t

        neighbors = _neighbors(indptr, indices, t)
        if len(neighbors) == 0:
            continue

        walk[walk_id, 1] = neighbors[
            np.searchsorted(_neighbors(indptr, data, t), np.random.rand())
        ]
        for j in range(2, walk_length):
            neighbors = _neighbors(indptr, indices, walk[walk_id, j - 1])
            if len(neighbors) == 0:
                break
            neighbors_weight = _neighbors(indptr, data, walk[walk_id, j - 1])
            if p == q == 1:
                # faster version
                walk[walk_id, j] = neighbors[
                    np.searchsorted(neighbors_weight, np.random.rand())
                ]
                continue
            while True:
                new_node = neighbors[
                    np.searchsorted(neighbors_weight, np.random.rand())
                ]
                r = np.random.rand()
                if new_node == walk[walk_id, j - 2]:
                    if r < prob_0:
                        break
                elif _isin_sorted(
                    _neighbors(indptr, indices, walk[walk_id, j - 2]), new_node
                ):
                    if r < prob_1:
                        break
                elif r < prob_2:
                    break
            walk[walk_id, j] = new_node
    return walk


@njit(nogil=True)
def _csr_row_cumsum(indptr, data):
    out = np.empty_like(data)
    for i in range(len(indptr) - 1):
        acc = 0
        for j in range(indptr[i], indptr[i + 1]):
            acc += data[j]
            out[j] = acc
        out[j] = 1.0
    return out


@njit(nogil=True)
def get_shortest_path(Pr, i, j):
    path = [j]
    k = j
    for it in range(100):
        if Pr[i, k] == -9999:
            break
        path.append(Pr[i, k])
        k = Pr[i, k]
    return path[::-1]