# mds20_replearning
This is the repository for the Models of Sequence Data 2020 Edition for the project Representation Learning with Contrastive Predictive Coding

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
Skip-thought vectors model is the alternative unsupervised approach for representation learning and the main competitor to the CPC model. We will use it as a baseline to estimate the performance of our CPC model implementation. 
Nevertheless, it is important to note that in the original paper by Oord A. CPC can't significantly beat Skip-thought model. 
https://github.com/sanyam5/skip-thoughts

## Extra materials 
Our notion page with additional information about project
https://www.notion.so/Representation-Learning-with-Contrastive-Predictive-Coding-85a758f444ab4757b5864ba248bcac75 
