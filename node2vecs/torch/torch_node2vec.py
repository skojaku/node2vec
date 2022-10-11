from node2vecs.node2vec import Node2Vec
import node2vecs


class TorchNode2Vec(Node2Vec):
    def __init__(self, device="cpu", batch_size=1024, **params):
        super().__init__(**params)
        self.device = device
        self.batch_size = batch_size

    def update_embedding(self, dim):

        # walks = self.sampler.sampling(n_walks = self.num_walks)

        # Word2Vec model
        model = node2vecs.Word2Vec(n_nodes=self.num_nodes, dim=dim)

        # Set up negative sampler
        dataset = node2vecs.NegativeSamplingDataset(
            seqs=self.sampler,
            window=self.window,
            epochs=self.epochs,
            context_window_type="double",
            num_negative_samples=self.negative,
            ns_exponent=self.ns_exponent,
        )

        # Set up the loss function
        loss_func = node2vecs.TripletLoss(model)

        # Train
        node2vecs.train(
            model=model,
            dataset=dataset,
            loss_func=loss_func,
            batch_size=self.batch_size,
            device=self.device,
            learning_rate=self.alpha,
        )
        self.in_vec = model.embedding()
        self.out_vec = model.embedding(return_out_vector=True)
