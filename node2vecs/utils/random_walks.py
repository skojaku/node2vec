# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2022-10-18 07:06:36
import numpy as np
from scipy import sparse
from numba import njit
from collections.abc import Iterable


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

    def __init__(self, adjmat, walk_length=40, p=1, q=1, padding_id=-1):
        """Random Walk Sampler.
        :param adjmat: Adjacency matrix of the graph.
        :type adjmat: scipy sparse matrix format (csr).
        :param walk_length: length per walk, defaults to 40
        :type walk_length: int, optional
        :param p: node2vec parameter p (1/p is the weights of the edge to previously visited node), defaults to 1
        :type p: float, optional
        :param q: node2vec parameter q (1/q) is the weights of the edges to nodes that are not directly connected to the previously visted node, defaults to 1
        :type q: float, optional
        """
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.padding_id = padding_id
        self.weighted = (~np.isclose(np.min(adjmat.data), 1)) or (
            ~np.isclose(np.max(adjmat.data), 1)
        )

        adjmat.sort_indices()
        self.indptr = adjmat.indptr.astype(np.int64)
        self.indices = adjmat.indices.astype(np.int64)
        if self.weighted:
            data = adjmat.data / adjmat.sum(axis=1).A1.repeat(np.diff(self.indptr))
            self.data = _csr_row_cumsum(self.indptr, data)

    def sampling(self, start):
        """Sample a random walk path.
        :param start: ID of the starting node
        :type start: int
        :return: array of visiting nodes
        :rtype: np.ndarray
        """
        if self.weighted:
            walk = _random_walk_weighted(
                self.indptr,
                self.indices,
                self.data,
                self.walk_length,
                self.p,
                self.q,
                self.padding_id,
                start
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
                self.padding_id,
                start
                if isinstance(start, Iterable)
                else np.array([start]).astype(np.int64),
            )
        if isinstance(start, Iterable):
            return walk.astype(np.int64)
        else:
            return walk[0].astype(np.int64)


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
            neighbors = _neighbors(indptr, indices, t)
            if len(neighbors) == 0:
                break
            neighbors_p = _neighbors(indptr, data, walk[walk_id, j - 1])
            if p == q == 1:
                # faster version
                walk[walk_id, j] = neighbors[
                    np.searchsorted(neighbors_p, np.random.rand())
                ]
                continue
            while True:
                new_node = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
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


#
# Homogenize the data format
#
def to_adjacency_matrix(net):
    """Convert to the adjacency matrix in form of sparse.csr_matrix.

    :param net: adjacency matrix
    :type net: np.ndarray or csr_matrix
    :return: adjacency matrix
    :rtype: sparse.csr_matrix
    """
    if sparse.issparse(net):
        if type(net) == "scipy.sparse.csr.csr_matrix":
            return net
        return sparse.csr_matrix(net)
    elif "numpy.ndarray" == type(net):
        return sparse.csr_matrix(net)
    else:
        ValueError("Unexpected data type {} for the adjacency matrix".format(type(net)))


def row_normalize(mat):
    """Normalize the matrix row-wise.

    :param mat: matrix
    :type mat: sparse.csr_matrix
    :return: row normalized matrix
    :rtype: sparse.csr_matrix
    """
    denom = np.array(mat.sum(axis=1)).reshape(-1).astype(float)
    return sparse.diags(1.0 / np.maximum(denom, 1e-32), format="csr") @ mat


def to_member_matrix(group_ids, node_ids=None, shape=None):
    """Create the binary member matrix U such that U[i,k] = 1 if i belongs to group k otherwise U[i,k]=0.

    :param group_ids: group membership of nodes. group_ids[i] indicates the ID (integer) of the group to which i belongs.
    :type group_ids: np.ndarray
    :param node_ids: IDs of the node. If not given, the node IDs are the index of `group_ids`, defaults to None.
    :type node_ids: np.ndarray, optional
    :param shape: Shape of the member matrix. If not given, (len(group_ids), max(group_ids) + 1), defaults to None
    :type shape: tuple, optional
    :return: Membership matrix
    :rtype: sparse.csr_matrix
    """
    if node_ids is None:
        node_ids = np.arange(len(group_ids))

    if shape is not None:
        Nr = int(np.max(node_ids) + 1)
        Nc = int(np.max(group_ids) + 1)
        shape = (Nr, Nc)
    U = sparse.csr_matrix(
        (np.ones_like(group_ids), (node_ids, group_ids)),
        shape=shape,
    )
    U.data = U.data * 0 + 1
    return U


def matrix_sum_power(A, T):
    """Take the sum of the powers of a matrix, i.e.,

    sum_{t=1} ^T A^t.

    :param A: Matrix to be powered
    :type A: np.ndarray
    :param T: Maximum order for the matrixpower
    :type T: int
    :return: Powered matrix
    :rtype: np.ndarray
    """
    At = np.eye(A.shape[0])
    As = np.zeros((A.shape[0], A.shape[0]))
    for _ in range(T):
        At = A @ At
        As += At
    return As


def pairing(k1, k2, unordered=False):
    """Cantor pairing function
    http://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function."""
    k12 = k1 + k2
    if unordered:
        return (k12 * (k12 + 1)) * 0.5 + np.minimum(k1, k2)
    else:
        return (k12 * (k12 + 1)) * 0.5 + k2


def depairing(z):
    """Inverse of Cantor pairing function http://en.wikipedia.org/wiki/Pairing_
    function#Inverting_the_Cantor_pairing_function."""
    w = np.floor((np.sqrt(8 * z + 1) - 1) * 0.5)
    t = (w**2 + w) * 0.5
    y = np.round(z - t).astype(np.int64)
    x = np.round(w - y).astype(np.int64)
    return x, y


def safe_log(A, minval=1e-12):
    if sparse.issparse(A):
        A.data = np.log(np.maximum(A.data, minval))
        return A
    else:
        return np.log(np.maximum(A, minval))


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


def csr_sampling(rows, csr_mat):
    return _csr_sampling(rows, csr_mat.indptr, csr_mat.indices, csr_mat.data)


@njit(nogil=True)
def _neighbors(indptr, indices_or_data, t):
    return indices_or_data[indptr[t] : indptr[t + 1]]


@njit(nogil=True)
def _csr_sampling(rows, indptr, indices, data):
    n = len(rows)
    retval = np.empty(n, dtype=indices.dtype)
    for j in range(n):
        neighbors = _neighbors(indptr, indices, rows[j])
        neighbors_p = _neighbors(indptr, data, rows[j])
        retval[j] = neighbors[np.searchsorted(neighbors_p, np.random.rand())]
    return retval
