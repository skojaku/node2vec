import unittest
import networkx as nx
import numpy as np
import gravlearn
import torch


class TestGravLearn(unittest.TestCase):
    def setUp(self):
        G = nx.karate_club_graph()
        self.A = nx.adjacency_matrix(G)
        self.labels = [G.nodes[i]["club"] for i in G.nodes]

        sampler = gravlearn.RandomWalkSampler(self.A, walk_length=40, p=1, q=1)
        self.seqs = [
            sampler.sampling(i) for _ in range(10) for i in range(self.A.shape[0])
        ]

    #    def test_word2vec_sampler(self):
    #        in_vec = np.random.randn(self.A.shape[0], 10)
    #        out_vec = np.random.randn(self.A.shape[0], 10)
    #        sampler = gravlearn.Word2VecSampler(in_vec, out_vec, alpha=1)
    #        sampler.fit(self.seqs)
    #        sampler.sampling()
    #        sampler.conditional_sampling(1)

    #    def test_precompressed_word2vec(self):
    #        device = "cpu"
    #        dim, base_dim = 32, 16
    #        base_emb = gravlearn.NormalizedLaplacianEmbedding(self.A, base_dim)
    #        model = gravlearn.PrecompressedWord2Vec(self.A.shape[0], dim, base_emb)
    #        dist_metric = gravlearn.DistanceMetrics.COSINE
    #        model = gravlearn.train(
    #            model,
    #            self.seqs,
    #            window_length=5,
    #            dist_metric=dist_metric,
    #            device=device,
    #        )
    #        emb = model.forward(torch.tensor(np.arange(self.A.shape[0])))
    #
    def test_grav_learn(self):
        device = "cpu"
        dim, base_dim = 32, 256
        # base_emb = gravlearn.fastRP(self.A, base_dim, 10, 1)
        # model = gravlearn.GravLearnModel(self.A.shape[0], dim, base_emb)
        model = gravlearn.Word2Vec(self.A.shape[0], dim)
        dist_metric = gravlearn.DistanceMetrics.COSINE
        model = gravlearn.train(
            model,
            self.seqs,
            # bags=self.A,
            window_length=1,
            dist_metric=dist_metric,
            device=device,
        )


if __name__ == "__main__":
    unittest.main()
