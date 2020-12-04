import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Subset, DataLoader

from skip_thought_vectors.dataloader import BookCorpusDataset


class BookCorpusDataModule(pl.LightningDataModule):
    def __init__(self, path, batch_size=128, sentences=None, word_dict=None,
                 valid_split=0):
        super(BookCorpusDataModule, self).__init__()
        self.path = path
        self.batch_size = batch_size
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
        self.dataset = BookCorpusDataset(self.path, sentences=self.sentences, word_dict=self.word_dict)
        _idx = np.arange(len(self.dataset))
        self._train_idx = _idx[:-self.valid_split*len(self.dataset)]
        self.train_ds = Subset(self.dataset, self._train_idx)
        self._valid_idx = np.setdiff1d(_idx, self._train_idx)
        self.valid_ds = Subset(self.dataset, self._valid_idx)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=False)

    def valid_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False)
