import mds20_replearning.skip_thought_vectors.utility as utility
import mds20_replearning.skip_thought_vectors.text_process_utility as text_process_utility
from torch.utils.data import Dataset
import torch


config = utility.read_config_file()
USE_CUDA = config["use_cuda"]
CUDA_DEVICE = config["cuda_device"]


class BookCorpusDataset(Dataset):
    EOS = 0  # to mean end of sentence
    UNK = 1  # to mean unknown token

    def __init__(self, text_file, sentences=None, word_dict=None):

        self.config = utility.read_config_file()

        with open(text_file, "rt") as f:
            sentences = f.readlines()
            word_dictionary = text_process_utility.build_dictionary(sentences)

        self.sentences = sentences
        self.word_dictionary = word_dictionary

        self.sentences_lens = [len(s) for s in self.sentences]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        ind = text_process_utility.encode_sentence(sent, self.word_dictionary,
                                                   self.config["vocabulary_dim"], self.config["max_len_for_rnn"],
                                                   self.EOS, self.UNK)
        length = min(len(sent.split()), self.config["max_len_for_rnn"])
        length = torch.LongTensor(length).cuda(CUDA_DEVICE)
        return ind, length
