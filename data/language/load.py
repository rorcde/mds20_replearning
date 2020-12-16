import os
import numpy as np


def load_data(path, name, return_textidxs=False):
    """
    Load one of MR, CR, SUBJ or MPQA
    """
    z = {}
    if name == 'rt-polarity':
        pos, neg = load_rt(path)
    elif name == 'Subjectivity_datasets':
        pos, neg = load_subj(path)
    elif name == 'review_polarity':
        if return_textidxs:
            pos, neg, pos_text_idxs, neg_text_idxs = load_polarity(path, return_textidxs)
        else:
            pos, neg = load_polarity(path, return_textidxs)

    labels = compute_labels(pos, neg)
    text = pos + neg

    text_idxs = None
    if return_textidxs and name == 'review_polarity':
        pos_len = len(pos)
        text_idxs = pos_text_idxs + [i+pos_len for i in neg_text_idxs]
        return text, labels, text_idxs
    else:
        return text, labels


def load_rt(loc = './dataset/Sentiment_polarity_datasets/rt_polaritydata/rt-polaritydata/'):
    """
    Load the rt-polarity dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'rt-polarity.pos'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'rt-polarity.neg'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_subj(loc='./dataset/Subjectivity_datasets/rotten_imdb/'):
    """
    Load the Subjectivity_datasets dataset
    """
    pos, neg = [], []
    with open(os.path.join(loc, 'plot.tok.gt9.5000'), 'rb') as f:
        for line in f:
            pos.append(line.decode('latin-1').strip())
    with open(os.path.join(loc, 'quote.tok.gt9.5000'), 'rb') as f:
        for line in f:
            neg.append(line.decode('latin-1').strip())
    return pos, neg


def load_polarity(loc='./dataset/Sentiment_polarity_datasets/review_polarity/txt_sentoken/', return_textidxs=False):
    """
    Load the review_polarity dataset
    """
    pos, neg = [], []

    temp_path = os.path.join(loc, 'pos')

    pos_text_idxs = [0]
    for file in os.listdir(temp_path):
        with open(os.path.join(temp_path, file), 'rb') as f:
            for line in f:
                text = line.decode("utf-8").strip()
                if len(text) > 0:
                    pos.append(text)
        pos_text_idxs.append(len(pos))

    temp_path = os.path.join(loc, 'neg')
    neg_text_idxs = [0]
    for file in os.listdir(temp_path):
        with open(os.path.join(temp_path, file), 'rb') as f:
            for line in f:
                text = line.decode("utf-8").strip()
                if len(text) > 0:
                    neg.append(text)
        neg_text_idxs.append(len(neg))

    if return_textidxs:
        return pos, neg, pos_text_idxs, neg_text_idxs
    else:
        return pos, neg


def compute_labels(pos, neg):
    """
    Construct list of labels
    """
    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1
    labels[len(pos):] = 0
    return labels


