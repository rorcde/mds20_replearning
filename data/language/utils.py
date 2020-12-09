from collections import Counter

from nltk import wordpunct_tokenize

import torch
from torch.utils import data
import numpy as np

from tqdm import tqdm


def build_vocabulary(sentences, vocab_size=None, pad_token='[PAD]', unk_token='[UNK]', eos_token='[EOS]'):
    counter = Counter()
    for s in tqdm(sentences):
        counter.update(wordpunct_tokenize(s))
    all_words = [pad_token, unk_token, eos_token] + [x[0] for x in counter.most_common(vocab_size)]
    w2i_mapping = {v: k for k, v in enumerate(all_words[:vocab_size])}
    return w2i_mapping


class DefaultDataset(data.Dataset):
    def __init__(self, raw_text, labels=None, vocab_size=None, w2i_mapping=None, pad_token='[PAD]',
                 unk_token='[UNK]', eos_token='[EOS]'):
        self.sentences = raw_text

        if w2i_mapping is not None:
            self.w2i_mapping = w2i_mapping
            self.vocab_size = len(w2i_mapping)
        else:
            self.w2i_mapping = build_vocabulary(self.sentences, vocab_size, pad_token, unk_token, eos_token)
            self.vocab_size = vocab_size

        self.pad_idx = self.w2i_mapping[pad_token]
        self.unk_idx = self.w2i_mapping[unk_token]
        self.eos_idx = self.w2i_mapping[eos_token]
        self.labels = labels
        self.vocab_size = len(self.w2i_mapping)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return [self.w2i_mapping.get(w, self.unk_idx) for w in wordpunct_tokenize(self.sentences[i])], \
               self.labels[i] if self.labels is not None else np.nan


def default_collate_fn(batch):
    text_batch = [b[0] for b in batch]
    labels_batch = [b[1] for b in batch]

    max_length = max(map(len, text_batch))
    text_batch = np.stack([np.pad(x, ((max_length - len(x), 0), )) for x in text_batch])

    return torch.from_numpy(text_batch), torch.FloatTensor(labels_batch)


"""
dataset = DefaultDataset(raw_text)
dataloader = data.Dataloader(dataset, batch_size, collate_fn=utils.collate_fn)
"""