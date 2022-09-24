[![Unit Test & Deploy](https://github.com/skojaku/gravlearn/actions/workflows/main.yml/badge.svg)](https://github.com/skojaku/gravlearn/actions/workflows/main.yml)

```python
import networkx as nx
import gravlearn as gn
import torch

# Load data
G = nx.karate_club_graph()
A = nx.adjacency_matrix(G)
labels = [G.nodes[i]["club"] for i in G.nodes]

# Generate the sequence for demo
sampler = gn.RandomWalkSampler(A, walk_length=40, p=1, q=1)
walks = [sampler.sampling(i) for _ in range(10) for i in range(A.shape[0])]

# Training
model = gravlearn.Word2Vec(A.shape[0], 32) # Embedding based on set

dist_metric = gravlearn.DistanceMetrics.EUCLIDEAN
model = gravlearn.train(model, walks, device = device, bags =A ,window_length=5, dist_metric=dist_metric)

# Embedding
emb = model.forward(torch.arange(A.shape[0]))
```
