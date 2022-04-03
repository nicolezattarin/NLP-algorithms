import re
from collections import Counter
import numpy as np

class Preprocess():
    """
    Preprocess text data.
    given a tring of text first apply cleaning function to remove punctuation, 
    then apply subsampling to remove words that appear less than threshold times.

    example:
        text = "Hi, I'm a sentence. This is another sentence."
        preprocess = Preprocess()
        words = preprocess.cleaning(text)
        words = preprocess.subsampling(words)
    """
    def __init__(self):
        pass

    def cleaning(self, text, cutoff=5):
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
        
        # Remove all words with  5 or fewer occurences
        word_counts = Counter(words)
        trimmed_words = [word for word in words if word_counts[word] > 5]
        return trimmed_words

    def create_lookup_tables(self, words):
        """
        Create lookup tables for vocabulary
        args: 
            words: a list of words from the vocabulary
        returns:
            vocab_to_int: dictionary that maps each word in the vocabulary to an integer (e.g. {'the':0})
            int_to_vocab: dictionary that maps each integer to a word in the vocabulary (e.g. {0:'the'})
        """
        word_counts = Counter(words)
        # sorting the words from most to least frequent in text occurrence
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        # dictionaries to convert words to integers and back again:
        # The integers are assigned in descending frequency order
        int_to_vocab = {i: word for i, word in enumerate(sorted_vocab)}
        vocab_to_int = {word: i for i, word in int_to_vocab.items()}

        return vocab_to_int, int_to_vocab

    def _text_to_int(self, words):
        """
        Convert text to series of integers: each integer represents a word in the vocabulary
        args:
            words: list of words
            vocab_to_int: dictionary
        returns:
            int_text: list of integers
        """
        vocab_to_int, int_to_vocab = self.create_lookup_tables(words)
        int_words = [vocab_to_int[word] for word in words]
        return int_words

    def subsampling (self, words, threshold=1e-5):
        """
        Subsampling words 
        args:
            words: list of words
            threshold: float (default: 1e-5) - remove words that occur less than threshold times
        returns:
            words: list of words
        """
    
        int_words = self._text_to_int(words)
        word_counts = Counter(int_words)
        # to reduce the size of the vocabulary and to reduce the noise in the data
        # we are only keeping words that appear more than threshold times
        tot_count = len(int_words)
        freq = {word: count/tot_count for word, count in word_counts.items()}
        drop_prob = {word: 1-np.sqrt(threshold/freq[word]) for word in word_counts}
        # discard some frequent words, according to the subsampling equation
        # For each word w_i in the training set, we'll discard it with probability given by 
        # P(w_i) = 1 - \sqrt{\frac{t}{f(w_i)}} 
        import random
        words = [word for word in int_words if random.random() < (1 - drop_prob[word])]
        return words
    
    def frequencies (self, words):
        """
        Calculate the frequency of each word in the vocabulary
        args:
            words: list of words
        returns:
            freq: dictionary of word frequencies
        """
        int_words = self._text_to_int(words)
        word_counts = Counter(int_words)
        tot_count = len(int_words)
        freq = {word: count/tot_count for word, count in word_counts.items()}
        return freq