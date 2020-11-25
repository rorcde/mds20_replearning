from nltk import word_tokenize

import torch
from torch.utils import data
import numpy as np


def get_tokenized_text(text):
    tokenized_text = []
    for sentence in text:
        tokenized_sentence = []
        for token in word_tokenize(sentence):
            tokenized_sentence.append(token)
        tokenized_text.append(tokenized_sentence)
    return tokenized_text


def build_vocabulary(tokenized_text, pad_token='[PAD]', unk_token='[PAD]', eos_token='[EOS]'):
    all_words = [pad_token, unk_token, eos_token] + list(set([x for sentence in tokenized_text for x in sentence]))
    w2i_mapping = {v: k for k, v in enumerate(all_words)}
    return w2i_mapping


class PosNegDataset(data.Dataset):
    def __init__(self, raw_text, labels, pad_token='[PAD]', unk_token='[PAD]', eos_token='[EOS]'):
        self.tokenized_text = get_tokenized_text(raw_text)
        self.w2i_mapping = build_vocabulary(self.tokenized_text, pad_token, unk_token, eos_token)
        self.labels = labels

    def __getitem__(self, i):
        return [self.w2i_mapping[w] for w in self.tokenized_text[i]], self.labels[i]


def collate_fn(batch):
    text_batch = [b[0] for b in batch]
    labels_batch = [b[1] for b in batch]

    max_length = max(map(len, text_batch))
    text_batch = np.stack([np.pad(x, ((max_length - len(x), 0), )) for x in text_batch])

    return torch.from_numpy(text_batch), torch.LongTensor(labels_batch)


"""
dataset = PosNegDataset(raw_text)
dataloader = data.Dataloader(dataset, batch_size, collate_fn=utils.collate_fn)
"""



