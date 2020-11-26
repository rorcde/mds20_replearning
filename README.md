# mds20_replearning
This is the repository for the Models of Sequence Data 2020 Edition for the project Representation Learning with Contrastive Predictive Coding. 

## Overview 
Creating good feature representation is one of the most core objectives in deep learning. The popular approach to deal with it is to use contrastive learning with Mutual Information (MI) as an objective function. One possible implementation of this is suggested in the paper "Representation Learning with Contrastive Predictive Coding".  In this work, we investigate the proposed model for several NLP  tasks and try to improve it using modern achievements of deep learning.

## CPC model 

### Our progress

Some preliminary versions of our code are already available! To launch the experiment, one need to do several simple steps:

```python
import sys
sys.path.append('path_to_folder_where_repo_lies')
sys.path.append('path_to_folder_where_repo_lies'+'mds20_replearning')

import pytorch_lightning as pl
from mds20_replearning.scripts.pl_data import PosNegDataModule
from mds20_replearning.scripts.pl_model import CPCModel
from mds20_replearning.data.language.load import load_polarity

model = CPCModel(len(dataset.w2i_mapping), emb_dim=620, enc_dim=2400, ar_dim=2400, kernel_size=5, lr=1e-3) 
data_module = PosNegDataModule(128, load_polarity, 'path_to_polarity_data', 
                               w2i_mapping=predefined_vocab_mapping, valid_split=0.3)

trainer = pl.Trainer()  # set required resources here
trainer.fit(model, datamodule=data_module)
```

## Previous implementations

### CPC model
We found dozens of implementations of CPC for images and audio, e.g.:
* https://github.com/Spijkervet/contrastive-predictive-coding - implementation for audio data (Pytorch)
* https://github.com/pat-coady/contrast-pred-code - implementation for audio data (TensorFlow)
* https://github.com/mf1024/Contrastive-Predictive-Coding-for-Image-Recognition-in-PyTorch - implementaton for images data (Pytorch)
* https://github.com/flrngel/cpc-tensorflow - implementaton for images data (TensorFlow)

However, there is only one Pytorch implementation related to natural language task.
* https://github.com/vgaraujov/CPC-NLP-PyTorch

The main part of the last one is [models.py](https://github.com/vgaraujov/CPC-NLP-PyTorch/blob/master/model/models.py). 
It implements convolutional encoder and model to setup experiment: Encoder united with GRU layer to create context embedding. 
The implementation of InfoNCE is unclear, however, it is easy to correct it.

The [utils folder](https://github.com/vgaraujov/CPC-NLP-PyTorch/tree/master/utils) has all functions necessary to build experiment. 
The [dataset.py](https://github.com/vgaraujov/CPC-NLP-PyTorch/blob/master/utils/dataset.py) creates two datasets "BookCorpus" and "SentimentAnalysis". It should be refactored before adaptation for MovieReview dataset.

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
