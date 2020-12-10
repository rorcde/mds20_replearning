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


class SkipThoughtEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim=620,  weights=None, encoder_dim=2400, pad_idx=0):
        super(SkipThoughtEmbedding).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim

        if weights is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        else:
            self.embedding = nn.Embedding.from_pretrained(weights)  # place for pre-trained embeddings

    def forward(self, input_sentences):

        word_embeddings = self.embedding(input_sentences)
        word_embeddings = torch.tanh(word_embeddings)

        return word_embeddings


class SkipThoughtEncoder(nn.Module):
    def __init__(self, embedding, batch_first=True):
        super(SkipThoughtEncoder).__init__()

        self.embedding = embedding

        self.rnn = nn.GRU(self.embedding.embedding_dim, self.embedding.encoder_dim,
                          bidirectional=False, batch_first=batch_first)

    def initialize_parameters(self):
        """
        Initialize all recurrent matricies with orthogonal initialization
        Initialize non-recurrent weights from a uniform distribution in [-0.1,0.1].
        """
        orthogonal_initialization(self.rnn)
        nn.init.uniform_(self.embedding.weight.data, -0.1, 0.1)

    def forward(self, input_sentences):
        if self.batch_first:
            input_sentences = torch.transpose(input_sentences, 0, 1)

        word_embeddings = self.embedding(input_sentences)
        output, h = self.rnn(word_embeddings)
        thought_vectors = h[-1]

        return thought_vectors, word_embeddings
