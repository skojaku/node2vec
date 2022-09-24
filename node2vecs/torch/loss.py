import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
#
# Distance metric
#
class DistanceMetrics(Enum):
    """
    The metric for the loasses
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    ANGULAR = lambda x, y: torch.arccos((1 - 1e-2) * F.cosine_similarity(x, y))
    DOTSIM = lambda x, y: -(x * y).sum(dim=1)

    def is_scale_invariant(dist_metric):
        return torch.isclose(
            dist_metric(torch.ones(1, 2), torch.ones(1, 2)),
            dist_metric(torch.ones(1, 2), 2 * torch.ones(1, 2)),
        )

#
# Loss function
#
class TripletLoss(nn.Module):
    def __init__(self, model, dist_metric=DistanceMetrics.DOTSIM, use_sigmoid=True, null_prob = None):
        super(TripletLoss, self).__init__()
        self.model = model
        self.weights = None
        self.dist_func = dist_metric
        self.logsigmoid = nn.LogSigmoid()
        self.use_sigmoid = use_sigmoid
        self.null_prob = null_prob

    def forward(self, iword, oword, onword):
        ivectors = self.model.forward_i(iword)
        ovectors = self.model.forward_o(oword)
        onvectors = self.model.forward_o(onword)

        oloss = -self.dist_func(ivectors, ovectors)
        nloss = -self.dist_func(ivectors, onvectors)
        if self.use_sigmoid:
            oloss = self.logsigmoid(oloss)
            nloss = self.logsigmoid(nloss.neg())
            return -(oloss + nloss).mean()
        else:
            return -(oloss - nloss + torch.pow(nloss, 2)/(self.null_prob[iword] * self.null_prob[oword]) ).mean()
            #return (-2 * oloss + torch.pow(nloss, 2)).mean()