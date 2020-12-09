from typing import Callable, Union

import pytorch_lightning as pl
from pathlib import Path
import numpy as np
from torch.utils.data import Subset, DataLoader
from torchvision.datasets.utils import download_and_extract_archive

from data.language.utils import DefaultDataset, default_collate_fn


def _default_load_func(data_path):  # suitable for bookcorpus
    with open(data_path, 'r') as fp:
        return '\n'.join(fp.readlines()).split('\n'), None


class DefaultDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                 data_path: Union[str, Path] = None,
                 load_func: Callable = _default_load_func,
                 url=None,
                 download_root=None,
                 filename=None,
                 vocab_size=20000,
                 w2i_mapping=None,
                 shuffle=False,
                 valid_split=0):
        super(DefaultDataModule, self).__init__()
        self.batch_size = batch_size
        self.data_path = Path(data_path)

        self.load_func = load_func
        self.url = url
        self.download_root = download_root
        self.filename = filename

        self.w2i_mapping = w2i_mapping
        self.valid_split = valid_split
        self.vocab_size = vocab_size
        self.shuffle = shuffle

        # defined later in setup
        self.dataset = None
        self._train_idx = None
        self._valid_idx = None
        self.train_ds = None
        self.valid_ds = None

    def prepare_data(self):
        if not self.data_path.exists():
            download_and_extract_archive(url=self.url, download_root=self.download_root)
        self.raw_text, self.labels = self.load_func(self.data_path)

    def _get_dataset(self):
        return DefaultDataset(self.raw_text, labels=self.labels, w2i_mapping=self.w2i_mapping, vocab_size=self.vocab_size)

    def setup(self, stage=None):
        self.dataset = self._get_dataset()
        _idx = np.arange(len(self.dataset))
        self._train_idx = _idx[:-int(self.valid_split*len(self.dataset))]
        self.train_ds = Subset(self.dataset, self._train_idx)
        self._valid_idx = np.setdiff1d(_idx, self._train_idx)
        self.valid_ds = Subset(self.dataset, self._valid_idx)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=default_collate_fn, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, collate_fn=default_collate_fn, shuffle=False)
