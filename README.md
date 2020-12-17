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

## Implementations

To train models one need to run corresponding notebook in the folder [notebooks](https://github.com/rodrigorivera/mds20_replearning/tree/master/notebooks) 
* train_cpc.ipyng - the CPC model training
* train_cpc.ipyng - the STV model training
* train_linear_model_mr_cpc.ipynb - evaluate CPC representation via linear classification
* train_linear_model_mr_stv.ipynb - evaluate STV representation via linear classification

We provide access to extra files via the link https://drive.google.com/drive/folders/1OAhVCmDprK598DbD83S-LvYHVPzscCMh?usp=sharing

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
