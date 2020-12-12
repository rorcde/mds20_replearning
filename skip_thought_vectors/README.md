### Skip-thought vectors model
Skip-thought vectors model is the alternative unsupervised approach for representation learning and the main competitor to the CPC model [[2]](#2). We will use it as a baseline to estimate the performance of our CPC model implementation. 
Nevertheless, it is important to note that in the original paper by Oord A [[1]](#1). CPC can't significantly beat Skip-thought model. 

We will use this implementation as a starting point for own model. 
* https://github.com/sanyam5/skip-thoughts

1. The most significant improvement is bringing the code in accordance with the article (using paper's embedding dimensions, using GRU instead of LSTM, initializing weights properly)
2. We also optimize some general solutions with respect to the best practices (there were unnecessary procedures and deprecated methods). 
3. Besides, we use some tricks: use left-padding instead of reverse sentences for encoder, add selection of the maximum length with respect to batch instead of setting one parameters for all sentences. 

## References
<a id="1">[1]</a> 
Oord A., Li Y., Vinyals O. 
Representation learning with contrastive predictive coding
arXiv preprint arXiv:1807.03748. – 2018.

<a id="2">[2]</a> 
Kiros R. et al. 
Skip-thought vectors
Advances in neural information processing systems. – 2015. – С. 3294-3302.
