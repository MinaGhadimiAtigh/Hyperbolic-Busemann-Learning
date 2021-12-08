import math

import torch
from torch import nn


################################################################################
# Penalty version
class PeBusePenalty(nn.Module):
    def __init__(self, dimension, penalty_option='dim', mult=1.0):
        super(PeBusePenalty, self).__init__()
        self.dimension = dimension
        if penalty_option == 'non':
            self.penalty_constant = 1.0
        elif penalty_option == 'dim':
            self.penalty_constant = mult * self.dimension
        else:
            print('~~~~~~~~!Your option is not available, I am choosing!~~~~~~~~')
            self.penalty_constant = 1.0

    def forward(self, p, g):
        # first part of loss
        prediction_difference = g - p
        difference_norm = torch.norm(prediction_difference, dim=1)
        difference_log = 2 * torch.log(difference_norm)

        # second part of loss
        data_norm = torch.norm(p, dim=1)
        proto_difference = (1 - data_norm.pow(2) + 1e-6)
        proto_log = (1 + self.penalty_constant) * torch.log(proto_difference)

        # second part of loss
        constant_loss = self.penalty_constant * math.log(2)

        one_loss = difference_log - proto_log + constant_loss
        total_loss = torch.mean(one_loss)

        return total_loss


################################################################################
class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cos_loss = nn.CosineSimilarity(eps=1e-9).cuda()

    def forward(self, p, g):
        return (1 - self.cos_loss(p, g)).pow(2).sum()


################################################################################
def buse_distance(p, g):
    data_norm = torch.norm(p, dim=1)
    denom = (1 - data_norm.pow(2) + 1e-6)

    prediction_difference = g - p

    numero = torch.norm(prediction_difference, dim=1)

    division = numero / denom

    one_loss = 2 * torch.log(division)
    total_loss = torch.mean(one_loss)

    return total_loss


def buse_distance_array(embedding1, embedding2):
    embedding2 = embedding2[:, None, :]
    data_norm = torch.norm(embedding1, dim=1)
    denom = (1 - data_norm.pow(2) + 1e-6)

    prediction_difference = embedding2 - embedding1
    numero = torch.norm(prediction_difference, dim=2)

    division = torch.div(torch.pow(numero, 2), denom)

    one_loss = torch.log(division)
    # one_loss = -1 * one_loss
    one_loss = torch.transpose(one_loss, 1, 0)

    return one_loss


def poincare_distance(u, v):
    diff = u - v

    u_norm = torch.norm(u, dim=1)
    v_norm = torch.norm(v, dim=1)
    diff_norm = torch.norm(diff, dim=1)

    return torch.acosh(1 + 2 * (diff_norm.pow(2) / ((1 - u_norm.pow(2)) * (1 - v_norm.pow(2)))))


def euclidean_dist(u, v):
    # return np.linalg.norm(u - v)
    return torch.cdist(u, v)
