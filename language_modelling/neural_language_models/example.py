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
text = text[:2000]

# test training
from FFNN import FFNeuralModel
embed_dim = 100
hidden_dim = 100
NLM = FFNeuralModel (embed_dim, hidden_dim, window_size=1)
NLM.train(text)

# loss history
loss = NLM.loss_history()
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(loss)
fig.savefig('loss.png')

# predict
print('\n\ntest prediction')
print(NLM.predict("free society"))

