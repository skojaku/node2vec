[![Unit Test & Deploy](https://github.com/skojaku/node2vecs/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/node2vecs/actions/workflows/main.yml)

# Gensim Implementation 

Gensim implementation is by far the most common and fastest form of the implementation of node2vec.

```python
TBD
```

# Torch Implementation 

Torch implementation offers flexibility and control over the training. 

```python
import node2vecs
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = np.unique([d[1]['club'] for d in G.nodes(data=True)], return_inverse=True)[1]

n_nodes = A.shape[0]
dim = 32

sampler = node2vecs.RandomWalkSampler(A, walk_length = 80)
walks = sampler.sampling(n_walks = 10)

# Word2Vec model
model = node2vecs.Word2Vec(n_nodes = n_nodes, dim = dim)

# Set up negative sampler
dataset = node2vecs.NegativeSamplingDataset(
    seqs=walks,
    window_length=10,
    epochs=1,
    context_window_type="double",
)

# Set up the loss function
loss_func = node2vecs.TripletLoss(model)

# Train
node2vecs.train(model = model, dataset = dataset, loss_func = loss_func, batch_size = 10000, device="cpu")
emb = model.embedding()
```
