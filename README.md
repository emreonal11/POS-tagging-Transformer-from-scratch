# Part of Speech Tagging Transformer from Scratch
This repo contains an implementation of an encoder-only transformer model for part-of-speech tagging. We have implemented this from scratch in both Pytorch and Matlab (coming soon). The most important part of the code is the implementation of the transformer backpropagation from scratch.

## Dataset
To run POS taggin on the conll 2003 dataset, first download the data:
* [Training set](https://drive.google.com/file/d/1PTfU4nI6aKrV9xsASFbOUf6Lkwxo1eD9/view?usp=sharing)
* [Testing set](https://drive.google.com/file/d/1RS4QIIv6TpCfden6bONfC1I4YqsJsBjA/view?usp=sharing)
* [Validation set](https://drive.google.com/file/d/1pkBoTOc1VE9kqGeGsjq57AgOAqjU6f0M/view?usp=sharing)

We use word2vec word embeddings which you can downlaod from here:
* [word vectors](https://drive.google.com/file/d/1v4VAsPCz6vqXrDqcF91i0okUnxZN3W_H/view?usp=sharing)

## Results
Number of Parameters : 202351 

Training Accuracy : 93.59%

Testing Accuracy : 89.62%
