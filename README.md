[![Unit Test & Deploy](https://github.com/skojaku/node2vecs/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/node2vecs/actions/workflows/main.yml)

# Install 

```bash 
git clone https://github.com/skojaku/node2vec.git
cd node2vec
pip install -e .
```


# Node2Vecs


## Gensim Implementation

Gensim implementation is by far the most common and fastest implementation of node2vec.

```python
import node2vecs
dim = 32 # embedding dimension
model = node2vecs.GensimNode2Vec(vector_size = dim)
model.fit(A) # adjacency matrix in scipy csr_matrix format
emb = model.transform()
```

## Torch Implementation

Torch implementation of node2vec, with hyperparameters aligned to those of gensim as much as possible.

```python
import node2vecs
dim = 32 # embedding dimension
model = node2vecs.TorchNode2Vec(vector_size = dim)
model.fit(A) # adjacency matrix in scipy csr_matrix format
emb = model.transform()
```

Torch implementation gives more control over the calculations.

```python
import node2vecs
import numpy as np

n_nodes = A.shape[0]
dim = 32


# Specify the sentense generator. 
# The sentence generator has ``sampling'' method which 
# generates a sequence of nodes for training word2vec.
sampler = node2vecs.RandomWalkSampler(A, walk_length = 80)

# Negative node sampler used in negative sampling
# The default is ConfigModelNodeSampler which samples a node 
# with probability proportional to it's degree.
# Changing noise sampler is useful to debias embedding. 
# utils.node_sampler has different noise sampler.
# See residual2vec paper.
noise_sampler = node2vecs.ConfigModelNodeSampler(ns_exponent=1.0)
noise_sampler.fit(A)

# Word2Vec model
model = node2vecs.Word2Vec(vocab_size = n_nodes, embedding_size= dim, padding_idx = n_nodes)

# Loss function
loss_func = node2vecs.Node2VecTripletLoss(n_neg=1)

# Set up negative sampler
dataset = node2vecs.TripletDataset(
    adjmat=A,
    window_length=10,
    noise_sampler=noise_sampler,
    padding_id=n_nodes,
    buffer_size=1e4,
    context_window_type="double", # we can limit the window to cover either side of center words. `context_window_type="double"` specifies a context window that extends both left and right of a focal node. context_window_type="left" or ="right" specifies that the window extends left or right, respectively.
    epochs=5, # number of epochs
    negative=1, # number of negative node per context
    p = 1, # (inverse) weight for the probability of backtracking walks 
    q = 1, # (inverse) weight for the probability of depth-first walks 
    walk_length = 80 # Length of walks
)

# Set up the loss function
loss_func = node2vecs.TripletLoss(model)

# Train
node2vecs.train(
    model=model,
    dataset=dataset,
    loss_func=loss_func,
    batch_size=256 * 4,
    device="cpu", # gpu is also available
    learning_rate=1e-3,
    num_workers=4,
)
model.eval()
in_vec = model.ivectors.weight.data.cpu().numpy()[:n_nodes, :] # in_vector
out_vec = model.ovectors.weight.data.cpu().numpy()[:n_nodes, :] # out_vector
```

## PytorchGeometric Implementation

```python
import node2vecs
dim = 32 # embedding dimension
model = node2vecs.PYGNode2Vec(vector_size = dim)
model.fit(A) # adjacency matrix in scipy csr_matrix format
emb = model.transform()
```


## Wrapper for other repositories

### node2vec

### fastnode2vec
