import os
import re
import sys
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.util import ngrams
from nltk import word_tokenize
from itertools import chain
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as torchdata

with open('test.txt') as f:
        text = f.read()
text = text
# test preprocessing
from FFNN import preprocessing, FFNN
ppmodel = preprocessing(text)
print("tokenized_text ", (ppmodel.tokenized_text[:10]))
print("hot_vect_corpus ", (ppmodel.get_hot_vect_corpus()[:4]))
print("vocab dim ", (ppmodel.vocab_dimension()))
vocab_dim = ppmodel.vocab_dimension()


#test network
embed_dim = 100
hidden_dim = 100
N=4      
model = FFNN(vocab_dim, embed_dim, hidden_dim, 3)
print(model)

# test training
from FFNN import FFNeuralModel
NLM = FFNeuralModel(vocab_dim, embed_dim, hidden_dim, window_size=3)

# dataset
data = ppmodel.get_hot_vect_corpus()
# data = ppmodel.get_tokenized()
NLM.train(data)
# v,l = NLM._prepare_data(data, 3)
# print("words: ",len(v[0]))
# print("labels: ", len(l[0]))