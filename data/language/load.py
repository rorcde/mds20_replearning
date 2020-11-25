import os
import numpy as np


def load_data(name, seed=1234):
    """
    Load one of MR, CR, SUBJ or MPQA
    """
    z = {}
    if name == 'rt-polarity':
        pos, neg = load_rt()
    elif name == 'Subjectivity_datasets':
        pos, neg = load_subj()
    elif name == 'review_polarity':
        pos, neg = load_polarity()
#     elif name == 'MPQA':
#         pos, neg = load_mpqa(loc=loc)

    labels = compute_labels(pos, neg)
    #text, labels = un_shuffle_data(pos+neg, labels, seed=seed)
    text = pos + neg
    z['text'] = text
    z['labels'] = labels
#     print("Type of text is {}".format(text))
#     features = encoder.encode(text)

    return z#, features


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


def load_polarity(loc='./dataset/Sentiment_polarity_datasets/review_polarity/txt_sentoken/'):
    """
    Load the review_polarity dataset
    """
    pos, neg = [], []
    temp_path = os.path.join(loc, 'pos')
    for file in os.listdir(temp_path):
        with open(os.path.join(temp_path, file), 'rb') as f:
            for line in f:
                text = line.decode("utf-8").strip()
                if len(text) > 0:
                    pos.append(text)
    temp_path = os.path.join(loc, 'neg')
    for file in os.listdir(temp_path):
        with open(os.path.join(temp_path, file), 'rb') as f:
            for line in f:
                text = line.decode("utf-8").strip()
                if len(text) > 0:
                    neg.append(text)
    return pos, neg


def compute_labels(pos, neg):
    """
    Construct list of labels
    """
    labels = np.zeros(len(pos) + len(neg))
    labels[:len(pos)] = 1
    labels[len(pos):] = 0
    return labels


