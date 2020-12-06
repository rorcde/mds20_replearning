from data.language.load import *
from pathlib import Path

import pytorch_lightning as pl
from typing import Union, Callable
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader, Subset
from data.language.utils import PosNegDataset, collate_fn
from itertools import chain


class PosNegDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, load_func: Callable, data_path: Union[str, Path], url=None, w2i_mapping=None,
                 valid_split=0):
        super(PosNegDataModule, self).__init__()
        self.batch_size = batch_size
        self.load_func = load_func
        self.data_path = Path(data_path)
        self.url = url
        self.w2i_mapping = w2i_mapping

        self.valid_split = valid_split

        # defined later in setup
        self.dataset = None
        self._train_idx = None
        self._valid_idx = None
        self.train_ds = None
        self.valid_ds = None

    def prepare_data(self):
        # download
        if not self.data_path.exists():
            download_and_extract_archive(url=self.url, extract_root=self.data_path)

    def setup(self, stage=None):
        pos, neg = self.load_func(self.data_path)
        labels = compute_labels(pos, neg)
        self.dataset = PosNegDataset(pos+neg, labels, w2i_mapping=self.w2i_mapping)
        if self.w2i_mapping is None:
            self.w2i_mapping = self.dataset.w2i_mapping

        train_pos_idx = np.arange(len(pos))[:-int(len(pos)*self.valid_split)]
        train_neg_idx = np.arange(len(pos), len(pos) + len(neg))[:-int(len(pos)*self.valid_split)]
        self._train_idx = np.concatenate((train_pos_idx, train_neg_idx))
        self.train_ds = Subset(self.dataset, self._train_idx)
        self._valid_idx = np.setdiff1d(np.arange(len(pos)+len(neg)), self._train_idx)
        self.valid_ds = Subset(self.dataset, self._valid_idx)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=False)

    def valid_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, collate_fn=collate_fn, shuffle=False)
