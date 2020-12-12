# mds20_replearning
This is the repository for the Models of Sequence Data 2020 Edition for the project Representation Learning with Contrastive Predictive Coding. 

## Overview 
Creating good feature representation is one of the most core objectives in deep learning. The popular approach to deal with it is to use contrastive learning with Mutual Information (MI) as an objective function. One possible implementation of this is suggested in the paper "Representation Learning with Contrastive Predictive Coding".  In this work, we investigate the proposed model for several NLP  tasks and try to improve it using modern achievements of deep learning.

The repository has the following structure:

---
| Path  | Description
| :---  | :----------
| [classification](https://github.com/rodrigorivera/mds20_replearning/tree/master/classification) | modules for evaluating representation via classification task
| [cpc](https://github.com/rodrigorivera/mds20_replearning/tree/master/cpc) | our implementation of CPC model
| data | utility for loading and preprocessing data
| &boxvr;&nbsp; imgs | technical folder
| &boxvr;&nbsp; [language](https://github.com/rodrigorivera/mds20_replearning/tree/master/data/language) | utility to preprocess NLP data, custom Dataset for NLP data 
| &boxvr;&nbsp; notebooks | notebooks for loading and observing considered datasets
| &boxvr;&nbsp; transaction | utility to preprocess transaction data, custom Dataset for transaction data (*in progress*)
| implementations | existing implementation for cpc model and the baseline (stv model) 
| &boxvr;&nbsp; [cpc_model_for_nlp](https://github.com/rodrigorivera/mds20_replearning/tree/master/implementations/cpc_model_for_nlp) | Pytorch implementation of the CPC model for NLP data
| &boxvr;&nbsp; [baseline_skip_thoughts_vectors](https://github.com/rodrigorivera/mds20_replearning/tree/master/implementations/baseline_skip_thoughts_vectors) | Pytorch implementatio for Skip-Thought vectors (model to compare with CPC)
| [notebooks](https://github.com/rodrigorivera/mds20_replearning/tree/master/notebooks) | notebooks for training models
| [scripts](https://github.com/rodrigorivera/mds20_replearning/tree/master/notebooks) | scripts for training via Pythorch Lighning
| &boxvr;&nbsp; cpc | scripts for cpc training via Pythorch Lighning
| &boxvr;&nbsp; skip_thoughts | scripts for stv training via Pythorch Lighning
| [skip_thought_vectors](https://github.com/rodrigorivera/mds20_replearning/tree/master/skip_thought_vectors) | our implementation of STV model

## CPC model 

### Our progress

Some preliminary versions of our code are already available! To launch the experiment, one need to do several simple steps:

```python
import sys
sys.path.append('/Users/vs/')  # your repo with mds20_replearning
sys.path.append('/Users/vs/'+'mds20_replearning')   # same

import pytorch_lightning as pl
from mds20_replearning.scripts.pl_data import DefaultDataModule
from mds20_replearning.scripts.cpc.pl_model import CPCModel
from mds20_replearning.data.language.load import load_polarity

import nltk
nltk.download('punkt')  # download if haven't yet

vocab_size = 20000

model = CPCModel(vocab_size, emb_dim=620, enc_dim=2400, ar_dim=2400, kernel_size=5, lr=2e-4) 
bookcorpus = DefaultDataModule(128, "/Users/vs/mds20_replearning/data/language/part_book_corpus.txt", valid_split=0.1, vocab_size=vocab_size)  # path to bookcorpus data

trainer = pl.Trainer()  # set required resources here
trainer.fit(model, datamodule=bookcorpus)
```

## Previous implementations

### Skip-thought vectors model
Skip-thought vectors model is the alternative unsupervised approach for representation learning and the main competitor to the CPC model [[2]](#2). We will use it as a baseline to estimate the performance of our CPC model implementation. 
Nevertheless, it is important to note that in the original paper by Oord A [[1]](#1). CPC can't significantly beat Skip-thought model. 

We will use this implementation as a starting point for own model. 
* https://github.com/sanyam5/skip-thoughts

1. The most significant improvement is bringing the code in accordance with the article (using paper's embedding dimensions, using GRU instead of LSTM, initializing weights properly)
2. Besides, we improve some general solutions with respect to the best practices (putting duplicate actions in functions, optimizating procedures e.t.c)
3. And the last one, but not least. We provide vocabulary expension with respect to article [[2]](#2). This part is missing in the implementation. 

## References
<a id="1">[1]</a> 
Oord A., Li Y., Vinyals O. 
Representation learning with contrastive predictive coding
arXiv preprint arXiv:1807.03748. – 2018.

<a id="2">[2]</a> 
Kiros R. et al. 
Skip-thought vectors
Advances in neural information processing systems. – 2015. – С. 3294-3302.

## Extra materials 
Our notion page with additional information about project
https://www.notion.so/Representation-Learning-with-Contrastive-Predictive-Coding-85a758f444ab4757b5864ba248bcac75 
