# mds20_replearning
This is the repository for the Models of Sequence Data 2020 Edition for the project Representation Learning with Contrastive Predictive Coding

## Implementations

We found dozens of implementations of CPC for images and audio, but only one for natural language.
* https://github.com/vgaraujov/CPC-NLP-PyTorch

The main part of the latter repository is [models.py](https://github.com/vgaraujov/CPC-NLP-PyTorch/blob/master/model/models.py). 
It implements convolutional encoder and model to setup experiment: Encoder united with GRU layer to create context embedding. 
The implementation of InfoNCE is unclear, however, it is easy to correct it.

The [utils folder](https://github.com/vgaraujov/CPC-NLP-PyTorch/tree/master/utils) has all functions necessary to build experiment. 
The [dataset.py](https://github.com/vgaraujov/CPC-NLP-PyTorch/blob/master/utils/dataset.py) creates two datasets "BookCorpus" and "SentimentAnalysis". It should be refactored before adaptation for MovieReview dataset.

