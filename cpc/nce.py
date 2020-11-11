import torch
from torch import nn
from torch.nn.functional import log_softmax


class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()
        pass

    def forward(self, positive, negative):
        probs = log_softmax(torch.stack(positive, negative, -1), -1)  # TODO: there is a simplier formula that doesn't require to compute all probabilities
        return probs[...:-1]
