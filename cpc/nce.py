import torch
from torch.nn.functional import log_softmax


def info_nce(predicted_samples, encoded_samples):
    """
        predicted_samples - next sequences, predicted from context representation
        encoded_samples - golden "next" sentences, passed through encoder

        returns mean info-nce loss per batch
    """

    # (batch, seq_len, enc_dim) x (batch, seq_len, enc_dim).T-> (batch, batch)
    scores = torch.bmm(predicted_samples,
                       encoded_samples.transpose(-2, -1))

    """
        Scores is a matrix: s_{ij} is a dot-product of i-th predicted sample and j-th sample from random samples.
        E_i is a positive sample for P_i; E_j, j != i, is considered as a negative sample for P_i
    """
    logprobs = log_softmax(scores, -1)
    return -torch.diagonal(logprobs, dim1=-2, dim2=-1).mean()
