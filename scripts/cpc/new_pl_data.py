from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from data.language.utils import build_vocabulary, get_tokenized_text
from itertools import chain
import numpy as np
import torch


class BookCorpusDataset(Dataset):
    EOS = 0  # to mean end of sentence
    UNK = 1  # to mean unknown token

    def __init__(self, text_file, sentences=None, word_dict=None, vocab_size=None,
                 pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]'):

        with open(text_file, "rt") as f:
            sentences = f.readlines()
            word_dictionary = build_vocabulary(sentences, vocab_size=vocab_size, pad_token=pad_token,
                                               unk_token=unk_token, eos_token=eos_token)

        self.word_dictionary = word_dictionary
            
        self.pad_idx = self.word_dictionary[pad_token]
        self.unk_idx = self.word_dictionary[unk_token]
        self.eos_idx = self.word_dictionary[eos_token]
        self.vocab_size = len(self.word_dictionary)

        self.tokenized_text = sentences         

    def __len__(self):
        return len(self.tokenized_text)

    def __getitem__(self, idx):
        return [self.word_dictionary.get(w, self.unk_idx) for w in self.tokenized_text[idx]]


def collate_fn(batch):
    max_length = max(map(len, batch))
    text_batch = np.stack([np.pad(x, ((max_length - len(x), 0), )) for x in batch])
    return torch.from_numpy(text_batch)


class BookCorpusDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=128, sentences=None, word_dict=None,
                 valid_split=0):
        
        super(BookCorpusDataModule, self).__init__()
        self.batch_size = batch_size
        self.data_path = Path(data_path)

        self.sentences = sentences
        self.word_dict = word_dict
        self.valid_split = valid_split

        # defined later in setup
        self.dataset = None
        self._train_idx = None
        self._valid_idx = None
        self.train_ds = None
        self.valid_ds = None

    def prepare_data(self):
        # load bookcorpus here
        pass

    def setup(self, stage=None):
        self.dataset = BookCorpusDataset(self.data_path, sentences=self.sentences, word_dict=self.word_dict)
        _idx = np.arange(len(self.dataset))
        self._train_idx = _idx[:-int(self.valid_split*len(self.dataset))]
        self.train_ds = Subset(self.dataset, self._train_idx)
        self._valid_idx = np.setdiff1d(_idx, self._train_idx)
        self.valid_ds = Subset(self.dataset, self._valid_idx)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=False)

    def valid_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=False)

