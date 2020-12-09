import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SkipThoughtEncoder(nn.Module):
    def __init__(self, **config):  # TODO: rename variable to agree with Datasets variables and CPC variables
        super().__init__()

        self.config = config
        self.vocabulary_dim = self.config["vocabulary_dim"]
        self.embedding_dim = self.config["embedding_dim"]
        self.thought_vectors_size = self.config["thought_vectors_size"]

        #
        self.rnn = nn.GRU(self.embedding_dim, self.thought_vectors_size, bidirectional=False)

    def initialize_parameters(self):
        """
        Initialize all recurrent matricies with orthogonal initialization
        Initialize non-recurrent weights from a uniform distribution in [-0.1,0.1].
        """
        orthogonal_initialization(self.rnn)
        nn.init.uniform_(self.embedding.weight.data, -0.1, 0.1)

    def forward(self, input_sentences):
        if self.config["batch_first"]:  # не меняй, так реально удобнее
            input_sentences = torch.transpose(input_sentences, 0, 1)

        word_embeddings = self.embedding(input_sentences)
        word_embeddings = torch.tanh(word_embeddings)

        output, h = self.rnn(word_embeddings)
        thought_vectors = h[-1]

        return thought_vectors, word_embeddings


class SkipThoughtDecoder(nn.Module):
    def __init__(self, **config): # TODO: rename variable to agree with Datasets variables and CPC variables
        super().__init__()

        self.config = config

        self.thought_vectors_size = self.config["thought_vectors_size"]
        self.embedding_dim = self.config["embedding_dim"]
        self.vocabulary_dim = self.config["vocabulary_dim"]

        self.rnn = nn.GRU(self.thought_vectors_size + self.embedding_dim, self.embedding_dim, bidirectional=False)
        self.output_layer = nn.Linear(self.embedding_dim, self.vocabulary_dim)

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
            ]
            , dim=-1)
        predicted_embeddings, _ = self.rnn(rnn_input)

        predicted_word = self.output_layer(predicted_embeddings)
        predicted_word = predicted_word.transpose(0, 1)

        return predicted_word


class SkipThoughtModel(nn.Module):
    def __init__(self, config):  # TODO: variables
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)  #TODO
        self.encoder = SkipThoughtEncoder()  #TODO: pass arguments
        self.decoder_next = SkipThoughtDecoder()  # TODO: pass arguments
        self.decoder_previous = SkipThoughtDecoder()  # TODO: pass arguments


    def initialize_parameters(self):
        self.encoder.initialize_parameters()
        self.decoder.initialize_parameters()

    def __calculate_loss(self, predictions, input_sentences, padding_idx=0):
        loss = F.cross_entropy(predictions, input_sentences, ignore_index=padding_idx)
        return loss

    def forward(self, input_sentences, *args):
        word_embeddings = self.embedding(input_sentences)
        thought_vectors, word_embeddings = self.encoder(word_embeddings)
        predicted_previous = self.decoder_previous(thought_vectors[1:], word_embeddings[:-1])  # TODO: check slices
        predicted_next = self.decoder_next(thought_vectors[:-1], word_embeddings[1:])

        loss_previous = self.__calculate_loss(predicted_previous, input_sentences[1:, :])
        loss_next = self.__calculate_loss(predicted_next, input_sentences[:-1, :])
        loss = loss_next + loss_previous

        _, predicted_previous_ids = predicted_previous[0].max(1)
        _, predicted_next_ids = predicted_next[0].max(1)

        return loss, loss_next, loss_previous, input_sentences[0], input_sentences[1], \
               predicted_previous_ids, predicted_next_ids
