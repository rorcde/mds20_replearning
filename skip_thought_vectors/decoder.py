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
    def __init__(self, encoder_dim, embedding_dim):
        super(SkipThoughtDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.embedding_dim = embedding_dim

        self.rnn = nn.GRU(self.encoder_dim + self.embedding_dim, self.embedding_dim,
                          bidirectional=False, batch_first=False)

    def initialize_parameters(self):
        """
        Initialize all recurrent matricies with orthogonal initialization
        Initialize non-recurrent weights from a uniform distribution in [-0.1,0.1].
        """
        orthogonal_initialization(self.rnn)

    def forward(self, thought_vectors, word_embeddings):
        seq_len = word_embeddings.shape[0]  # the time dimension is supposed to be the first one
        rnn_input = torch.cat(
            [
                word_embeddings,
                thought_vectors.unsqueeze(0).expand(seq_len, -1, -1)
            ],
            dim=-1)
        output, h = self.rnn(rnn_input)

        return output

class SkipThoughtOutput(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super(SkipThoughtOutput, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        self.linear = nn.Linear(self.embedding_dim, self.vocab_size)
        
    def initialize_parameters(self):
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        

    def forward(self, predicted_embedding):
        seq_len, batch_size, decoder_dim = predicted_embedding.size()
        predicted_embedding = self.linear(predicted_embedding.view(-1, decoder_dim)).view(seq_len, batch_size, -1)
        predicted_embedding = predicted_embedding.transpose(0, 1).contiguous()        
        return predicted_embedding