import torch.nn as nn
import torch.nn.functional as F
from .encoder import SkipThoughtEncoder, SkipThoughtEmbedding
from .decoder import SkipThoughtDecoder, SkipThoughtOutput
import torch

class SkipThoughtModel(nn.Module):
    def __init__(self,  vocab_size, embedding_dim=620,  weights=None, encoder_dim=2400, pad_idx=0, batch_first=True):
        super(SkipThoughtModel, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encoder_dim = encoder_dim
        self.batch_first = batch_first

        self.embedding = SkipThoughtEmbedding(self.vocab_size, self.embedding_dim, weights, self.encoder_dim, pad_idx, self.batch_first)
        self.encoder = SkipThoughtEncoder(self.embedding_dim, self.encoder_dim)
        self.decoder_next = SkipThoughtDecoder(self.encoder_dim, self.embedding_dim)
        self.decoder_previous = SkipThoughtDecoder(self.encoder_dim, self.embedding_dim)
        self.output_layer = SkipThoughtOutput(self.embedding_dim, self.vocab_size)
        

    def initialize_parameters(self):
        self.encoder.initialize_parameters()
        self.embedding.initialize_parameters()
        self.decoder_next.initialize_parameters()
        self.decoder_previous.initialize_parameters()
        self.output_layer.initialize_parameters()

        
    def __calculate_loss(self, predictions, input_sentences, padding_idx=0):
        loss = F.cross_entropy(predictions, input_sentences, ignore_index=padding_idx)
        return loss

    def forward(self, input_sentences, *args):
        #if self.batch_first:
        #    input_sentences = torch.transpose(input_sentences, 0, 1) 
        word_embeddings = self.embedding(input_sentences)
        thought_vectors = self.encoder(word_embeddings)
        predicted_previous = self.decoder_previous(thought_vectors[1:, :], word_embeddings[:, :-1, :])
        predicted_next = self.decoder_next(thought_vectors[:-1, :], word_embeddings[:, 1:, :])        
        predicted_previous = self.output_layer(predicted_previous)[:, :-1]
        predicted_next = self.output_layer(predicted_next)[:, :-1]

        input_sentences = torch.transpose(input_sentences, 0, 1) 
        loss_previous = self.__calculate_loss(predicted_previous.reshape(-1, self.vocab_size), input_sentences[1:, 1:].reshape(-1))
        loss_next = self.__calculate_loss(predicted_next.reshape(-1, self.vocab_size), input_sentences[:-1, 1:].reshape(-1))
        
        loss = loss_next + loss_previous

        _, predicted_previous_ids = predicted_previous[0].max(1)
        _, predicted_next_ids = predicted_next[0].max(1)

        return loss, input_sentences[0], input_sentences[1], \
               predicted_previous_ids, predicted_next_ids
