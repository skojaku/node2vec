[![Unit Test & Deploy](https://github.com/skojaku/node2vecs/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/node2vecs/actions/workflows/main.yml)

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

# Generate data to train word2vec
sampler = node2vecs.RandomWalkSampler(A, walk_length = 80)
walks = sampler.sampling(n_walks = 10)

# Word2Vec model
model = node2vecs.Word2Vec(n_nodes = n_nodes, dim = dim)

# Set up negative sampler
dataset = node2vecs.NegativeSamplingDataset(
    seqs=walks,
    window=10,
    epochs=1,
    context_window_type="double", # we can limit the window to cover either side of center words.
    ns_exponent = 1 # exponent of negative sampling
)

# Set up the loss function
loss_func = node2vecs.TripletLoss(model)

# Train
node2vecs.train(model = model, dataset = dataset, loss_func = loss_func, batch_size = 10000, device="cpu")
emb = model.embedding()
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
