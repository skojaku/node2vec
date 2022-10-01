from enum import Enum


class DefaultParams(Enum):
    """
    List of default parameter values
    """

    #
    # Pytorch geometric
    # refs:
    # 1. https://github.com/pyg-team/pytorch_geometric/blob/master/examples/node2vec.py
    # 2. https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.models.Node2Vec
    # For a parameter that takes different values in refs [1] and [2], we adopt the value in Ref. 2 since
    # that's the default value more people use.
    PYG = {
        "num_walks": 1,  # number of walkers per node
        "walk_length": 20,  # number of walks per walker
        "p": 1.0,  # bias parameter
        "q": 1.0,  # bias parameter
        "window": 10,  # context window size
        "vector_size": 128,  # embedding dimension
        "ns_exponent": 0,  # exponent for negative sampling
        "alpha": 0.01,  # learning rate
        "epochs": 100,  # epochs
        "negative": 1,  # number of negative samples per positive sample
    }

    #
    # Gensim (Aditya Grover)
    # refs:
    # 1. https://radimrehurek.com/gensim/models/word2vec.html
    # 2. https://github.com/aditya-grover/node2vec/blob/master/src/main.py
    #
    GENSIM = {
        "num_walks": 10,  # number of walkers per node
        "walk_length": 80,  # number of walks per walker
        "p": 1.0,  # bias parameter
        "q": 1.0,  # bias parameter
        "window": 10,  # context window size
        "vector_size": 128,  # embedding dimension
        "ns_exponent": 0.75,  # exponent for negative sampling
        "alpha": 0.025,  # learning rate
        "epochs": 1,  # epochs
        "negative": 5,  # number of negative samples per positive sample
    }

    #
    # Torch (Our implementation)
    #
    TORCH = {
        "num_walks": 10,  # number of walkers per node
        "walk_length": 80,  # number of walks per walker
        "p": 1.0,  # bias parameter
        "q": 1.0,  # bias parameter
        "window": 10,  # context window size
        "vector_size": 128,  # embedding dimension
        "ns_exponent": 0.75,  # exponent for negative sampling
        "alpha": 0.025,  # learning rate
        "epochs": 1,  # epochs
        "negative": 5,  # number of negative samples per positive sample
    }
