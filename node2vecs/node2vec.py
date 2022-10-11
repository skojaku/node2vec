from .utils.random_walks import RandomWalkSampler
from scipy import sparse
import numpy as np

#
# Base class
#
class Node2Vec:
    """node2vec implementation

    Parameters
    ----------
    num_walks : int (optional, default 10)
        Number of walks per node
    walk_length : int (optional, default 40)
        Length of walks
    window_length : int (optional, default 10)
    restart_prob : float (optional, default 0)
        Restart probability of a random walker.
    p : node2vec parameter (TODO: Write doc)
    q : node2vec parameter (TODO: Write doc)
    """

    def __init__(
        self,
        num_walks=10,  # number of walkers per node
        walk_length=80,  # number of walks per walker
        p=1.0,  # bias parameter
        q=1.0,  # bias parameter
        window=10,  # context window size
        vector_size=64,  # embedding dimension
        ns_exponent=0.75,  # exponent for negative sampling
        alpha=0.025,  # learning rate
        epochs=1,  # epochs
        negative=5,  # number of negative samples per positive sample
    ):
        self.in_vec = None  # In-vector
        self.out_vec = None  # Out-vector
        self.rw_params = {
            "p": p,
            "q": q,
            "walk_length": walk_length,
            "num_walks":num_walks,
        }
        self.ns_exponent = ns_exponent
        self.alpha = alpha
        self.epochs = epochs
        self.negative = negative
        self.num_walks = num_walks
        self.num_nodes = None
        self.vector_size = vector_size
        self.sentences = None
        self.model = None
        self.window = window

    def fit(self, net):
        """Estimating the parameters for embedding."""
        net = self.homogenize_net_data_type(net)
        self.num_nodes = net.shape[0]
        self.sampler = RandomWalkSampler(net, **self.rw_params)

    def transform(self, vector_size=None, return_out_vector=False):
        """Compute the coordinates of nodes in the embedding space of the
        prescribed dimensions."""
        # Update the in-vector and out-vector if
        # (i) this is the first to compute the vectors or
        # (ii) the dimension is different from that for the previous call of transform function
        if vector_size is None:
            vector_size = self.vector_size

        if self.out_vec is None:
            self.update_embedding(vector_size)
        elif self.out_vec.shape[1] != vector_size:
            self.update_embedding(vector_size)
        return self.out_vec if return_out_vector else self.in_vec

    def update_embedding(self, dim):
        # Update the dimension and train the model
        # Sample the sequence of nodes using a random walk
        pass

    def homogenize_net_data_type(self, net):
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
            ValueError(
                "Unexpected data type {} for the adjacency matrix".format(type(net))
            )
