
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
The implementation of InfoNCE is unclear, however, it is easy to correct it. The [utils folder](https://github.com/vgaraujov/CPC-NLP-PyTorch/tree/master/utils) has all functions necessary to build experiment. The [dataset.py](https://github.com/vgaraujov/CPC-NLP-PyTorch/blob/master/utils/dataset.py) creates two datasets "BookCorpus" and "SentimentAnalysis". It should be refactored before adaptation for MovieReview dataset.

This implementation is located in the folder cpc_model_for_nlp. There you can find more information and the instruction how to run the model.

### Skip-thought vectors model
We will use following implementation as a starting point for own model:
* https://github.com/sanyam5/skip-thoughts

This implementation is located in the folder  baseline_skip_thoughts_vectors. There you can find more information and the instruction how to run the model.
