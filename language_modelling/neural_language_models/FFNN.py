from audioop import bias
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

class preprocessing():
    def __init__(self, corpus):
        self.corpus = corpus
        self.clean_corpus = self._preprocess(corpus)
        self._tokenization(self.clean_corpus) # tokenize the corpus
        self.vocab_to_int, self.int_to_vocab = self.create_lookup_tables(self.tokenized_text)
        #create the corpus as a list of hot vectors i.e. vectors with 1 at the index of the word in the vocab,
        #and 0 elsewhere, accroding to the index of vocab_to_int
        self.HotVecCorpus = self._hot_vect_corpus(self.tokenized_text, self.vocab_to_int)

    def _tokenization (self, corpus):
        """
        Tokenize the corpus.
        """
        from nltk import word_tokenize, sent_tokenize 
        
        self.tokenized_text = word_tokenize(corpus)
    
    def _preprocess(self, text, cutoff=5):
        """
        Preprocess text for word2vec, replace punctuation with tokens so we can use them in our model
        args:
            text: str
            cutoff: int (default: 5) - remove words that appear less than cutoff times
        """

        # Replace punctuation with tokens so we can use them in our model
        text = text.lower()
        text = text.replace('.', ' <PERIOD> ')
        text = text.replace(',', ' <COMMA> ')
        text = text.replace('"', ' <QUOTATION_MARK> ')
        text = text.replace(';', ' <SEMICOLON> ')
        text = text.replace('!', ' <EXCLAMATION_MARK> ')
        text = text.replace('?', ' <QUESTION_MARK> ')
        text = text.replace('(', ' <LEFT_PAREN> ')
        text = text.replace(')', ' <RIGHT_PAREN> ')
        text = text.replace('--', ' <HYPHENS> ')
        text = text.replace('?', ' <QUESTION_MARK> ')
        text = text.replace('\n', ' <NEW_LINE> ')
        text = text.replace(':', ' <COLON> ')
        words = text.split()

        clean_corpus = ' '.join(words)
        return clean_corpus

    def create_lookup_tables(self, words):
        """
        Create lookup tables for vocabulary
        args: 
            words: a list of words from the vocabulary
        returns:
            vocab_to_int: dictionary that maps each word in the vocabulary to an integer (e.g. {'the':0})
            int_to_vocab: dictionary that maps each integer to a word in the vocabulary (e.g. {0:'the'})
        """
        from collections import Counter
        word_counts = Counter(words)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {i: word for i, word in enumerate(sorted_vocab)}
        vocab_to_int = {word: i for i, word in int_to_vocab.items()}
        return vocab_to_int, int_to_vocab
    
    def _hot_vect_corpus(self, words, vocab_to_int):
        """
        Create a hot vector representation of the corpus
        args:
            words: a list of words from the vocabulary
        returns:
            hot_vect_corpus: a list of hot vectors, one hot vector for each word in the corpus
        """
        hot_vect_corpus = []
        for w in words:
            hot_vect = np.zeros(len(vocab_to_int))
            hot_vect[self.vocab_to_int[w]] = 1
            hot_vect_corpus.append(hot_vect)
        return hot_vect_corpus

    def get_hot_vect_corpus(self):
        return self.HotVecCorpus

    def vocab_dimension(self):
        """
        Return the dimension of the vocabulary
        """
        self.VocabDim = len(self.vocab_to_int)
        return len(self.vocab_to_int)

    def get_preprocessed_corpus(self):
        """
        Return the preprocessed corpus
        """
        return self.clean_corpus
    
    def get_tokenized(self):
        return self.tokenized_text


class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, window_size=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lin1 = nn.Linear(embedding_dim*window_size, hidden_dim)
        self.activation = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, *w):
        # w is a list of words, as many as the window
        if self.window_size > 1:
            e = [self.embedding(torch.tensor(np.nonzero(w[i])[0][0]).long()) for i in range(len(w))]
            e = torch.cat(e, dim=-1)
        else:
            e = self.embedding(w)

        h = self.lin1(e)
        h = self.activation(h)
        z = self.lin2(h)
        y = self.softmax(z)
        return y
    
class FFNeuralModel():
    def __init__(self, vocab_size, embedding_dim, hidden_dim, window_size=3, lr=1e-2):
        self.model = FFNN(vocab_size, embedding_dim, hidden_dim, window_size=window_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_function = nn.CrossEntropyLoss()
        self.trained = False
        self.window_size = window_size
        print('Model initialized')


    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def _prepare_data(self, words):
        """
        prepare data for training
        """
        datatrain = []
        labels = []
        for i in range(len(words)-self.window_size):
            datatrain.append([words[i:i+self.window_size]])
            labels.append(words[i+self.window_size])
        print('Data prepared')
        return datatrain, labels
    
    def train(self, words, n_epochs=10, verbose=True, print_every=100):
        """
        args:
            words: list of words from the vocabulary as 1hot vectors
            n_epochs: int (default: 10) - number of epochs to train for
            verbose: bool (default: True) - whether to print progress
            print_every: int (default: 10) - how often to print progress
        """
        from tqdm import tqdm
        self.history = []

        # send the model to GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        words, labels = self._prepare_data(words)

        if verbose: iterable = range(n_epochs)
        else: iterable = tqdm(range(n_epochs), desc = "Training in progress")
        nsteps = 0
        for e in iterable:
            if verbose: print('Epoch {}'.format(e+1))
            # get our input, target batches
            for w, l in zip(words, labels):
                l = torch.tensor([np.nonzero(l)[0][0]]).long()
                self.optimizer.zero_grad()
                output = self.model(*w[0])
                copy_output = output.clone().tolist()
                loss = self.loss_function(torch.tensor([copy_output]).float(), l)
                loss.requires_grad = True
                loss.backward()
                self.optimizer.step()
                nsteps += 1
                if nsteps % print_every == 0:
                    self.history.append(loss.item())
                    if verbose:
                        print(f"Loss: {loss.item()}")
            print("\n")
        self.trained = True