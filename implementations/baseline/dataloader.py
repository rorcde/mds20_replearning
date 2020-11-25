import torch
from torch.autograd import Variable
from vocab import *
import numpy as np
import random
import utility

np.random.seed(0)


class DataLoader:
    EOS = 0  # to mean end of sentence
    UNK = 1  # to mean unknown token

    def __init__(self, text_file, sentences=None, word_dict=None):

        self.config = utility.read_config_file()

        with open(text_file, "rt") as f:
            sentences = f.readlines()
            word_dict = build_and_save_dictionary(sentences, source=text_file)

        self.sentences = sentences
        self.word_dict = word_dict
        print("Making reverse dictionary")
        self.revmap = list(self.word_dict.items())

        self.lengths = [len(sent) for sent in self.sentences]

    def convert_sentence_to_indices(self, sentence):

        indices = [
                      # assign an integer to each word, if the word is too rare assign unknown token
                      self.word_dict.get(w) if self.word_dict.get(w, VOCAB_SIZE + 1) < VOCAB_SIZE else self.UNK

                      for w in sentence.split()  # split into words on spaces
                  ][: self.maxlen - 1]  # take only maxlen-1 words per sentence at the most.

        # last words are EOS
        indices += [self.EOS] * (self.maxlen - len(indices))

        indices = np.array(indices)
        indices = Variable(torch.from_numpy(indices))
        if USE_CUDA:
            indices = indices.cuda(CUDA_DEVICE)

        return indices

    def convert_indices_to_sentences(self, indices):

        def convert_index_to_word(idx):

            idx = idx.item()
            if idx == 0:
                return "EOS"
            elif idx == 1:
                return "UNK"

            search_idx = idx - 2
            if search_idx >= len(self.revmap):
                return "NA"

            word, idx_ = self.revmap[search_idx]

            assert idx_ == idx
            return word

        words = [convert_index_to_word(idx) for idx in indices]

        return " ".join(words)

    def fetch_batch(self, batch_size):

        first_index = random.randint(0, len(self.sentences) - batch_size)
        batch = []
        lengths = []

        for i in range(first_index, first_index + batch_size):
            sent = self.sentences[i]
            ind = self.convert_sentence_to_indices(sent)
            batch.append(ind)
            lengths.append(min(len(sent.split()), MAXLEN))

        batch = torch.stack(batch)
        lengths = np.array(lengths)

        return batch, lengths
