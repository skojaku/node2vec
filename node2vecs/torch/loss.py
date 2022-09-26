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
    def __init__(self, model, dist_metric=DistanceMetrics.DOTSIM, with_logsigmoid=True):
        super(TripletLoss, self).__init__()
        self.model = model
        self.weights = None
        self.dist_func = dist_metric
        self.logsigmoid = nn.LogSigmoid()
        self.with_logsigmoid = with_logsigmoid

    def forward(self, iword, oword, y):
        ivectors = self.model.forward_i(iword)
        ovectors = self.model.forward_o(oword)

        loss = -self.dist_func(ivectors, ovectors) * y
        if self.with_logsigmoid:
            loss = self.logsigmoid(loss)
        return -(loss).mean()


class ModularityTripletLoss(TripletLoss):
    def __init__(self, **params):
        super(ModularityTripletLoss, self).__init__(**params)

    def forward(self, iword, oword, y, pcoef):
        ivectors = self.model.forward_i(iword)
        ovectors = self.model.forward_o(oword)

        loss = -self.dist_func(ivectors, ovectors) * y
        if self.with_logsigmoid:
            loss = self.logsigmoid(loss)
        loss = torch.pow(loss, pcoef)
        return -(loss).mean()