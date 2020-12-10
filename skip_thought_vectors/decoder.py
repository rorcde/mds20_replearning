import torch
import torch.nn as nn


def orthogonal_initialization(gru_cell, gain=1):
    # https://discuss.pytorch.org/t/initializing-rnn-gru-and-lstm-correctly/23605/2
    # https://gist.github.com/kaniblu/81828dfcf5cca60ae93f4d7bd19aeac5
    # https://pytorch.org/docs/stable/nn.init.html

    """
    Orthogonal initialization of recurrent weights
    GRU cell contains 3 matrices in one parameter, so we slice it.
    gain - optional scaling factor (default=1)
    """

    with torch.no_grad():
        for _, hh, _, _ in gru_cell.all_weights:
            for i in range(0, hh.size(0), gru_cell.hidden_size):
                nn.init.orthogonal_(hh.data[i: i + gru_cell.hidden_size], gain=gain)


class SkipThoughtDecoder(nn.Module):
    def __init__(self, embedding, batch_first):
        super(SkipThoughtDecoder).__init__()

        self.embedding = embedding
        self.encoder_dim = self.embedding.encoder_dim
        self.embedding_dim = self.embedding.embedding_dim
        self.vocab_size = self.embedding.vocab_size

        self.rnn = nn.GRU(self.encoder_dim + self.embedding_dim, self.embedding_dim,
                          bidirectional=False, batch_first=batch_first)

    def initialize_parameters(self):
        """
        Initialize all recurrent matricies with orthogonal initialization
        Initialize non-recurrent weights from a uniform distribution in [-0.1,0.1].
        """
        orthogonal_initialization(self.rnn)
        nn.init.uniform_(self.output_layer.weight.data, -0.1, 0.1)

    def forward(self, thought_vectors, word_embeddings):
        seq_len = word_embeddings.shape[0]  # the time dimension is supposed to be the first one
        rnn_input = torch.cat(
            [
                word_embeddings,
                thought_vectors.unsqueeze(0).expand(seq_len, -1, -1)
            ],
            dim=-1)
        _, h = self.rnn(rnn_input)

        predicted_word = torch.dot(word_embeddings, h)

        #predicted_word = self.output_layer(predicted_embeddings)
        #predicted_word = predicted_word.transpose(0, 1)

        return predicted_word
