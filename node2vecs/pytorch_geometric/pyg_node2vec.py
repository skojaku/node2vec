import numpy as np
from ..node2vec import Node2Vec
from tqdm import tqdm
import torch_geometric
import torch
from scipy import sparse


class PYGNode2Vec(Node2Vec):
    def __init__(self, device="cpu", **params):
        super().__init__(**params)
        self.device = device

    def fit(self, A):
        r, c, _ = sparse.find(A)
        self.edge_index = torch.tensor(np.vstack([r, c]), dtype=torch.long)
        self.num_nodes = A.shape[0]

    def update_embedding(self, dim):

        model = torch_geometric.nn.Node2Vec(
            self.edge_index,
            embedding_dim=dim,
            walk_length=self.rw_params["walk_length"],
            context_size=self.window,
            walks_per_node=self.num_walks,
            num_negative_samples=self.negative,
            p=self.rw_params["p"],
            q=self.rw_params["q"],
            sparse=True,
        ).to(self.device)
        model.train()  # put model in train model

        loader = model.loader(
            batch_size=128, shuffle=True, num_workers=4
        )  # data loader to speed the train
        optimizer = torch.optim.SparseAdam(
            list(model.parameters()), lr=self.alpha
        )  # initzialize the optimizer

        for _ in range(self.epochs):
            total_loss = 0
            for pos_rw, neg_rw in tqdm(loader):
                optimizer.zero_grad()  # set the gradients to 0
                loss = model.loss(
                    pos_rw.to(self.device), neg_rw.to(self.device)
                )  # compute the loss for the batch
                loss.backward()
                optimizer.step()  # optimize the parameters
                total_loss += loss.item()
            total_loss = total_loss / len(loader)

        emb = model(torch.arange(self.num_nodes, device=self.device))
        if self.device != "cpu":
            emb = emb.detach().cpu().numpy()
        else:
            emb = emb.detach().numpy()

        self.in_vec = emb
        self.out_vec = emb
