# mds20_replearning
This is the repository for the Models of Sequence Data 2020 Edition for the project Representation Learning with Contrastive Predictive Coding. 

## Overview 
Creating good feature representation is one of the most core objectives in deep learning. The popular approach to deal with it is to use contrastive learning with Mutual Information (MI) as an objective function. One possible implementation of this is suggested in the paper "Representation Learning with Contrastive Predictive Coding".  In this work, we investigate the proposed model for several NLP  tasks and try to improve it using modern achievements of deep learning.

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
