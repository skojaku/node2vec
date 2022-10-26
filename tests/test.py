import unittest
import networkx as nx
import numpy as np
import node2vecs

class TestNode2Vecs(unittest.TestCase):
    def setUp(self):
        G = nx.karate_club_graph()
        self.A = nx.to_scipy_sparse_array(G)
        self.labels = [G.nodes[i]["club"] for i in G.nodes]

    def test_torch_node2vec(self):

        dim = 32
        model = node2vecs.TorchNode2Vec(vector_size = dim)
        model.fit(self.A)
        emb = model.transform()
        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == dim

    def test_gensim_node2vec(self):
        dim = 32
        model = node2vecs.GensimNode2Vec(vector_size = dim)
        model.fit(self.A)
        emb = model.transform()
        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == dim

    def test_torch_geometric_node2vec(self):
        dim = 32
        model = node2vecs.PYGNode2Vec(vector_size = dim)
        model.fit(self.A)
        emb = model.transform()
        assert emb.shape[0] == self.A.shape[0]
        assert emb.shape[1] == dim

if __name__ == "__main__":
    unittest.main()
