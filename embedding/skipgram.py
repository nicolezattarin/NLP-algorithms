"""
This code is a rearrangement of the code from the following repository:
     https://github.com/udacity/deep-learning-v2-pytorch/tree/master/word2vec-embeddings

We rearranged the code in a class to perform an automatic preprocessing of the text
and to provide an easy interface to the skipgram model.
Moreover, we added functions for accessing the embeddings, the loss history 
and visualizing the embeddings.
"""

import torch
from torch import nn
import torch.optim as optim

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None, device='cpu'):
        """
        Args:
            n_vocab: Number of words in the vocabulary.
            n_embed: Number of dimensions to embed into.
            noise_dist: (Optional) A distribution over words to use as noise.
            device: Device to use for computation. Use 'cpu' for cpu and 'cuda' for gpu.
        """
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        self.device = device
        
        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # initialize the embedding weights uniformly
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors
    
    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    
    def forward_noise(self, batch_size, n_samples):
        """ 
        Generate noise vectors with shape (batch_size, n_samples, n_embed)
        """
        if self.noise_dist is None:
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)
        noise_words = noise_words.to(self.device)
        
        #.view(): tensor with the same data as the self tensor but of a different shape.
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        return noise_vectors

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        """
        Forward pass of the model.
        args:
            input_vectors: tensor of shape (batch_size, dim_embed)
            output_vectors: tensor of shape (batch_size, dim_embed)
            noise_vectors: tensor of shape (batch_size, n_samples, dim_embed)
        """
        #reshaping
        batch_size, dim_embed = input_vectors.shape
        input_vectors = input_vectors.view(batch_size, dim_embed, 1)
        output_vectors = output_vectors.view(batch_size, 1, dim_embed)
        
        # compute the scores: batch matrix-matrix product +sigmoid + log
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors
        return -(out_loss + noise_loss).mean()




class SkipGram():
    def __init__(self, corpus, dim_embed, noise_distribution=None, device="cpu", lr=0.001):
        """
        Initialize the SkipGram model.
        args:
            corpus: text corpus
            dim_embed: dimension of the embedding matrix
            noise_distribution: distribution of the noise words
            device: device to run the model on, default is cpu, but can be set to cuda
            lr: learning rate
        """

        self.preprocessing(corpus)
        self.model = SkipGramNeg(len(self.vocab_to_int), dim_embed, noise_distribution).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = NegativeSamplingLoss() 
        self.device = device
        self.lr = lr
        self.trained = False #if True, then the model has been trained
        print("SkipGram initialized")
        

    def cosine_similarity(self, embedding, valid_size=16, valid_window=100, device='cpu'):
        """ 
        Returns the cosine similarity of validation words with words in the embedding matrix.
        Here, embedding should be a PyTorch embedding module.
        args:
            embedding: embedding module
            valid_size: number of validation words to consider
            valid_window: number of words to consider left and right of the target
            device: device to run the model on, default is cpu, but can be set to cuda
        """
        
        # sim = (a . b) / |a||b|
        import numpy as np
        import random
        embed_vectors = embedding.weight #get the embedding matrix from the embedding module
        
        # magnitude of embedding vectors, |b|
        magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)
        
        # pick N words from our ranges (0,window) and (1000,1000+window). lower id implies more frequent 
        valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
        valid_examples = np.append(valid_examples,
                                random.sample(range(1000,1000+valid_window), valid_size//2))
        valid_examples = torch.LongTensor(valid_examples).to(device)
        
        # compute embedding
        valid_vectors = embedding(valid_examples)
        # compute cosine similarity
        similarities = torch.mm(valid_vectors, embed_vectors.t())/magnitudes
        
        return valid_examples, similarities

    def _window(self, words, idx, window_size=5):
        """
        Get a list of words (as integers) in a window around an index.
        args:
            words: list of words
            target_word: index of the center word
            window_size: number of words on each side of the center word
        """
        for w in words: 
            if not isinstance(w, int): raise ValueError("words must be integers")
        import numpy as np
        R = np.random.randint(1, window_size+1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = words[start:idx] + words[idx+1:stop+1]
        
        return list(target_words)

    def _get_batches(self, words, batch_size, window_size=5):
        """
        Create a generator of word batches as a tuple (inputs, targets)  
        args:   
            words: list of words
            batch_size: number of words per batch
            window_size: number of words on each side of the center word
         """
        
        n_batches = len(words)//batch_size
        words = words[:n_batches*batch_size]
        
        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx+batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = self._window(batch, ii, window_size)
                y.extend(batch_y)
                x.extend([batch_x]*len(batch_y))
            yield x, y
    
    def preprocessing (self, corpus):
        """
        Preprocesses the corpus by converting it to a list of integers
        and removing words that are not in the embedding matrix.
        args:
            corpus: text corpus
        """
        from preprocess import Preprocess
        import numpy as np
        pp = Preprocess()
        self.words = pp.cleaning(corpus)
        self.train_words = pp.subsampling(self.words)
        self.vocab_to_int, self.int_to_vocab = pp.create_lookup_tables(self.words)
        freqs = pp.frequencies(self.words)
        word_freqs = np.array(sorted(freqs.values(), reverse=True))
        unigram_dist = word_freqs/word_freqs.sum()
        self.noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

    def train(self,  n_epochs=10, batch_size=512, n_samples=5, verbose=False, print_every=100, window_size=5):
        """
        args:
            input_words: list of ints - input words
            output_words: list of ints - output words
            n_epochs: int - number of epochs to train for
            batch_size: int - size of the batch
            n_samples: int - number of negative samples to use
        """

        nsteps = 0
        from tqdm import tqdm
        import time
        self.history = []

        if verbose: iterable = range(n_epochs)
        else: iterable = tqdm(range(n_epochs), desc = "Training in progress")
        
        for e in iterable:
            # get our input, target batches
            for input_words, target_words in self._get_batches(self.train_words, batch_size, window_size):
                nsteps += 1
                inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # input, outpt, and noise vectors
                input_vectors = self.model.forward_input(inputs)
                output_vectors = self.model.forward_output(targets)
                noise_vectors = self.model.forward_noise(inputs.shape[0], n_samples)

                # negative sampling loss
                loss = self.criterion(input_vectors, output_vectors, noise_vectors)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if verbose:
                    if nsteps % print_every == 0:
                        print("Epoch: {}/{}".format(e+1, n_epochs))
                        print("Loss: ", loss.item())

            # save loss stats
            self.history.append(loss.item())
        self.trained = True
        print("SkipGram trained")
    
    def history_loss(self):
        """
        Returns the history of the loss
        """
        if self.trained:return self.history
        else:raise ValueError("Model has not been trained yet")
    
    def plot_loss(self, figsize=(10,6), **kwargs):
        """
        Plots the loss history
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig,ax=plt.subplots(figsize=figsize)

        sns.set_theme(style='white',palette='Dark2',font_scale=1.5)
        sns.lineplot(x=range(len(self.history)), y=self.history, **kwargs)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        fig.savefig("skipgram_loss.png", bbox_inches='tight')
    
    def embedding(self, word):
        """
        given a word, either as string or int, return the embedding vector
        Return a dictionary of word: embedding
        """
        if isinstance(word, str): index = self.vocab_to_int[word]
        else: index = word
        if self.trained: return {self.int_to_vocab[index]: self.embeddings()[index]}
        else:raise ValueError("Model has not been trained yet")

    def embeddings(self):
        """
        Returns the embedding matrix
        """
        if self.trained:return self.model.in_embed.weight.to(self.device).data.numpy()
        else:raise ValueError("Model has not been trained yet")
    
    def save_model(self, filename):
        """
        Saves the model to a file
        """
        if self.trained:
            torch.save(self.model.state_dict(), filename)
            print("Model saved to {}".format(filename))
        else:raise ValueError("Model has not been trained yet")


    def visualize_embed (self, n_words, path="embeddings.png", 
                            figsize=(10,10), 
                            title="Word Embeddings",
                            color = "darkred", fontsize=15, **kwargs):
        """
        Visualizes embedded words 
        args:
            n_words: number of words to plot
            path: to save image
        """

        if not self.trained: raise ValueError("Model has not been trained yet")
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import seaborn as sns
        sns.set_theme(style='white', font_scale=2.2)

        embeddings = self.embeddings()
        tsne = TSNE()
        embed_tsne = tsne.fit_transform(embeddings[:n_words, :])
        fig, ax = plt.subplots(figsize=figsize)
        for idx in range(n_words):
            plt.scatter(*embed_tsne[idx, :], color=color, **kwargs)
            plt.annotate(self.int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), fontsize=fontsize)
        ax.set_title (title)
        fig.savefig(path, bbox_inches='tight')

    def int_to_vocab(self, index):
        """
        given an index, return the word
        """
        return self.int_to_vocab[index]

    def vocab_to_int(self, word):
        """
        given a word, return the index
        """
        return self.vocab_to_int[word]