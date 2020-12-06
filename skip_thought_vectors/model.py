import torch
import torch.nn as nn
import torch.nn.functional as F
import utility


class SkipThoughtEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.config = utility.read_config_file()
        self.vocabulary_dim = self.config["vocabulary_dim"]
        self.embedding_dim = self.config["embedding_dim"]
        self.thought_vectors_size = self.config["thought_vectors_size"]

        self.embedding = nn.Embedding(self.vocabulary_dim, self.embedding_dim)
        self.rnn = nn.GRU(self.embedding_dim, self.thought_vectors_size, bidirectional=False)

    def initialize_parameters(self):
        """
        Initialize all recurrent matricies with orthogonal initialization
        Initialize non-recurrent weights from a uniform distribution in [-0.1,0.1].
        """
        utility.orthogonal_initialization(self.rnn)
        nn.init.uniform_(self.embedding.weight.data, -0.1, 0.1)

    def forward(self, input_sentences, sentences_lens):
        if self.config["batch_first"]:
            input_sentences = torch.transpose(input_sentences, 0, 1)

        word_embeddings = self.embedding(input_sentences)
        word_embeddings = torch.tanh(word_embeddings)

        reversed_word_embeddings = utility.reverse_variables(word_embeddings)
        output, h = self.rnn(reversed_word_embeddings)
        thought_vectors = h[-1]

        return thought_vectors, word_embeddings


class SkipThoughtDecoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.config = utility.read_config_file()

        self.thought_vectors_size = self.config["thought_vectors_size"]
        self.embedding_dim = self.config["embedding_dim"]
        self.vocabulary_dim = self.config["vocabulary_dim"]

        self.rnn_for_previous = nn.GRU(self.thought_vectors_size + self.embedding_dim,
                                       self.embedding_dim, bidirectional=False)
        self.rnn_for_next = nn.GRU(self.thought_vectors_size + self.embedding_dim,
                                   self.embedding_dim, bidirectional=False)
        self.output_layer = nn.Linear(self.embedding_dim, self.vocabulary_dim)

    def initialize_parameters(self):
        """
        Initialize all recurrent matricies with orthogonal initialization
        Initialize non-recurrent weights from a uniform distribution in [-0.1,0.1].
        """
        utility.orthogonal_initialization(self.rnn_for_previous)
        utility.orthogonal_initialization(self.rnn_for_next)
        nn.init.uniform_(self.output_layer.weight.data, -0.1, 0.1)

    def __forward_one_direction(self, thought_vectors, word_embeddings, predict_next=False):
        if predict_next:
            curr_thought_vectors = thought_vectors[:, 1:, :]
            curr_word_embedding = word_embeddings[:, 1:, :]
            rnn = self.rnn_for_next
        else:
            curr_thought_vectors = thought_vectors[:, :-1, :]
            curr_word_embedding = word_embeddings[:, :-1, :]
            rnn = self.rnn_for_previous

        expanded_curr_word_embedding = utility.expand_tensor(curr_word_embedding)
        rnn_input = torch.cat([curr_thought_vectors, expanded_curr_word_embedding], dim=2)
        predicted_embeddings, _ = rnn(rnn_input)

        seq_len, batch_size, embedding_size = predicted_embeddings.size()
        predicted_word = self.output_layer(predicted_embeddings.reshape(seq_len * batch_size, embedding_size))
        predicted_word = predicted_word.view(seq_len, batch_size, -1).transpose(0, 1).contiguous()

        return predicted_word

    def forward(self, thought_vectors, word_embeddings):
        # prepare thought vectors for rnn
        thought_vectors = thought_vectors.repeat(self.config["max_len_for_rnn"], 1, 1)

        predicted_previous = self.__forward_one_direction(thought_vectors, word_embeddings, predict_next=False)
        predicted_next = self.__forward_one_direction(thought_vectors, word_embeddings, predict_next=True)

        return predicted_previous, predicted_next


class SkipThoughtModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = SkipThoughtEncoder()
        self.decoder = SkipThoughtDecoder()
        self.config = self.encoder.config

    def initialize_parameters(self):
        self.encoder.initialize_parameters()
        self.decoder.initialize_parameters()

    def __calculate_loss(self, predictions, input_sentences, sentences_lens, predicted_next=False):
        if predicted_next:
            sentences_lens = sentences_lens[1:]
            input_sentences = input_sentences[1:, :]
        else:
            sentences_lens = sentences_lens[:-1]
            input_sentences = input_sentences[:-1, :]

        masked_predictions = predictions * utility.create_mask(predictions, sentences_lens)
        masked_predictions = masked_predictions.view(-1, self.config["vocabulary_dim"])
        loss = F.cross_entropy(masked_predictions, input_sentences.view(-1))
        return loss

    def forward(self, input_sentences, sentences_lens):

        thought_vectors, word_embeddings = self.encoder(input_sentences, sentences_lens)
        predicted_previous, predicted_next = self.decoder(thought_vectors, word_embeddings)

        loss_previous = self.__calculate_loss(predicted_previous, input_sentences, sentences_lens)
        loss_next = self.__calculate_loss(predicted_next, input_sentences, sentences_lens)
        loss = loss_next + loss_previous

        _, predicted_previous_ids = predicted_previous[0].max(1)
        _, predicted_next_ids = predicted_next[0].max(1)

        return loss, loss_next, loss_previous, input_sentences[0], input_sentences[1], predicted_previous_ids, predicted_next_ids


