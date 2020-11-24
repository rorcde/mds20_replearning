import torch
from torch import nn
from torch.nn.functional import log_softmax


class InfoNCE(nn.Module):
    def __init__(self):
        super(InfoNCE, self).__init__()
        pass

    @staticmethod
    def forward(predicted_samples, encoded_samples):
        """
        predicted_samples - next sequences, predicted from context representation
        encoded_samples - golden "next" sentences, passed through encoder

        returns mean info-nce loss per batch
        """

        scores = torch.mm(predicted_samples, encoded_samples.transpose(0, 1))  # (batch, enc_len) x (batch, enc_len) -> (batch, batch)

        """
        Scores is a matrix: s_{ij} is a dot-product of i-th predicted sample and j-th sample from random samples.
        E_i is a positive sample for P_i; E_j, j != i, is considered as a negative sample for P_i
        """
        logprobs = log_softmax(scores, 1)
        return torch.diagonal(logprobs).mean()
