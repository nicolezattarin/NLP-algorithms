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
        """
        Initialize the preprocessing class
        args:
            corpus: a string of text
        """
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
    
    def _preprocess(self, text):
        """
        Preprocess, replace punctuation with tokens so we can use them in our model
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
            vocab_to_int: dictionary that maps each word in the vocabulary to an integer (e.g. {'the':0})
        returns:
            hot_vect_corpus: a list of hot vectors, one hot vector for each word in the corpus
        """
        hot_vect_corpus = []
        for w in words:
            hot_vect = np.zeros(len(vocab_to_int))
            hot_vect[vocab_to_int[w]] = 1
            hot_vect_corpus.append(hot_vect)
        return hot_vect_corpus

    def get_hot_vect_corpus(self):
        """
        Return the hot vector representation of the corpus
        """
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
        """
        Return the tokenized corpus
        """
        return self.tokenized_text


class FFNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, window_size=1):
        """
        Initialize the FFNN model
        args:
            vocab_size: the size of the vocabulary
            embedding_dim: the dimension of the embedding layer
            hidden_dim: the dimension of the hidden layer
            window_size: the size of the window to observe the context of the word
        """
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
    
    def forward(self, w):
        """
        Forward pass of the model
        args:
            w: list of hot vectors, one hot vector for each word in the window
        returns:
            out: the output of the model
        """
        e = [self.embedding(torch.tensor(np.nonzero(w[i])[0][0]).long()) for i in range(len(w))]
        e = torch.cat(e, dim=-1)

        h = self.lin1(e)
        h = self.activation(h)
        z = self.lin2(h)
        y = self.softmax(z)
        return y
    
class FFNeuralModel():
    def __init__(self, embedding_dim, hidden_dim, window_size=1):
        """
        Initialize the FFNN model
        args:
            embedding_dim: the dimension of the embedding layer
            hidden_dim: the dimension of the hidden layer
            window_size: the size of the window to observe the context of the word
        """
        if embedding_dim < 1: raise ValueError('embedding_dim must be greater than 0')
        if hidden_dim < 1: raise ValueError('hidden_dim must be greater than 0')
        if window_size < 1: raise ValueError('window_size must be greater than 0')
        if not isinstance(embedding_dim, int): raise TypeError('embedding_dim must be an integer')
        if not isinstance(hidden_dim, int): raise TypeError('hidden_dim must be an integer')
        if not isinstance(window_size, int): raise TypeError('window_size must be an integer')

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.loss_function = nn.CrossEntropyLoss()
        self.trained = False
        self.window_size = window_size

    def save_model(self, path):
        """
        Save the model
        args:
            path: the path to save the model
        """
        torch.save(self.model.state_dict(), path)
    
    def _prepare_data(self, corpus):
        """
        prepare data for training
        """
        self.pp = preprocessing(corpus)
        self.onehotwords = self.pp.get_hot_vect_corpus()
        self.vocab_dim = self.pp.vocab_dimension()
        self.vocab_to_int, self.int_to_vocab = self.pp.vocab_to_int, self.pp.int_to_vocab

        # cretae data and labels
        datatrain = []
        labels = []
        for i in range(len(self.onehotwords)-self.window_size):
            datatrain.append([self.onehotwords[i:i+self.window_size]])
            labels.append(self.onehotwords[i+self.window_size])
        print('Data prepared')
        self.datatrain = datatrain
        self.labels = labels
    
    def train(self, corpus, n_epochs=3, 
                    verbose=True, print_every=100, 
                    lr=1e-3, momentum=0.9, 
                    save=True, save_path='model.pt'):
        """
        args:
            corpus: the corpus to train the model on
            n_epochs: the number of epochs to train the model
            verbose: whether to print the loss after each epoch
            print_every: the number of epochs to wait before printing the loss
            lr: the learning rate
            momentum: the momentum parameter for SGD
            save: whether to save the model
            save_path: the path to save the model
        """
        from tqdm import tqdm
        self.history = []

        # send the model to GPU or CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare data
        self._prepare_data(corpus)

        # create model
        self.model = FFNN(self.vocab_dim, self.embedding_dim, self.hidden_dim, self.window_size)
        print('Model initialized')
        print('Window size: ', self.window_size)
        print('Vocab size: ', self.vocab_dim)
        print('Embedding dimension: ', self.embedding_dim)
        print('Hidden dimension: ', self.hidden_dim)
        self.model.to(device)
        print('Model sent to device')

        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # train
        if verbose: iterable = range(n_epochs)
        else: iterable = tqdm(range(n_epochs), desc = "Training in progress")
        nsteps = 0
        for e in iterable:
            if verbose: print('Epoch {}'.format(e+1))
            running_loss = 0.0
            # get our input, target batches
            for w, l in zip(self.datatrain, self.labels):
                l = torch.tensor([np.nonzero(l)[0][0]]).long()
                self.optimizer.zero_grad()
                output = self.model(w[0])
                copy_output = output.clone().tolist()
                loss = self.loss_function(torch.tensor([copy_output]).float(), l)
                loss.requires_grad = True
                loss.backward()
                self.optimizer.step()
                nsteps += 1
                running_loss += loss.item()
                if nsteps % print_every == 0:   
                    print(f"loss: {loss.item():.3f}")
                    running_loss = 0.0
            print("\n")
            self.history.append(loss.item())
        self.trained = True
        print('Training finished')
        # save model
        if save:self.save_model(save_path)

    def loss_history(self):
        """
        return the loss history
        """
        if self.trained: return self.history
        else: raise Exception('Model not trained')

    def predict(self, sentence):
        """
        args:
            sentence: string of words
        """
        if not self.trained: raise Exception('Model not trained')
        # debug: up to now we support only words which are in the vocabulary
        # get hot vector
        words = sentence.split(' ')
        hotvects = []
        for w in words:
            hot_vect = np.zeros(len(self.vocab_to_int))
            hot_vect[self.vocab_to_int[w]] = 1
            hotvects.append(hot_vect)
            output = self.model([hotvects])
        return self.int_to_vocab[np.argmax(output.detach().numpy())]
