from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten

class NGrams():
    """
    example:
    >>> from ngram import NGrams
    >>> ngram = NGrams(corpus, 2)   
    >>> ngram.train()
    >>> ngram.generate(10)
    >>> # compute probability of a word given a context
    >>> ngram.probability('the', 'a') # probability of 'the' given 'a'
    """
    
    def __init__(self, corpus, n):
        """
        args:
            corpus: string of words
            n: integer of ngram length
        """
        self.corpus = corpus
        self.n = n
        self.tokenization()
        self.get_words_list() 
        self.ngram_generation()
        self.trained = False

    def tokenization (self):
        """
        Tokenize the corpus.
        """
        try: # Use the default NLTK tokenizer.
            from nltk import word_tokenize, sent_tokenize 
            # Testing whether it works. 
            word_tokenize(sent_tokenize("This is a foobar sentence. Yes it is.")[0])
        except: # Use a naive sentence tokenizer and toktok.
            import re
            from nltk.tokenize import ToktokTokenizer
            # See https://stackoverflow.com/a/25736515/610569
            sent_tokenize = lambda x: re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', x)
            toktok = ToktokTokenizer()
            word_tokenize = word_tokenize = toktok.tokenize

        self.tokenized_text = [list(map(str.lower, word_tokenize(sent))) 
                  for sent in sent_tokenize(self.corpus)]

    def get_words_list(self):
        """
        Returns a list of words from the corpus.
        """
        self.words_list = self.corpus.split()
        return self.words_list 

    def ngram_generation(self):
        """
        Generate ngrams from a corpus.
        """
        from nltk.lm.preprocessing import padded_everygram_pipeline
        self.train_data, self.padded_sents = padded_everygram_pipeline(self.n, self.tokenized_text)

    def train(self):
        """
        Train the model.
        """
        from nltk.lm import MLE
        self.estimator = MLE(self.n)
        self.estimator.fit(self.train_data, self.padded_sents)
        self.trained = True

    def count(self, word, *context):
        """
        Return the number of times a given word appears in the corpus.
        if mpre than a single word is given, return the number of times the first word is seed in the
        context of the second word.

        args:
            word: string of word
            context: serie of words (optional) that precede the word
        """
        if not self.trained: raise ValueError("Model not trained yet.")
        if not context: return self.estimator.counts[word]
        else: return self.estimator.counts[list(context)][word]

    def probability (self, word, *context):
        """
        score how probable is a given word given a context.
        """
        if not self.trained: raise ValueError("Model not trained yet.")
        
        if not context: return self.estimator.score(word)
        else: self.estimator.score(word, context)

    def log_probability (self, word, *context):
        """
        score how probable is a given word given a context.
        """
        if not self.trained: raise ValueError("Model not trained yet.")
        if not context: return self.estimator.score(word)
        else: self.estimator.logscore(word, context)

    def generate(self, nwords=1, text_seed=None, random_seed=None, detokenize=True):
        """
        Generate a text from the model.
        args:
            nwords: number of words to generate
            text_seed: Generation can be conditioned on preceding context.
            random_seed: Random seed for the generation.
            detokenize: Whether to detokenize the generated text.
        """
        if not self.trained: raise ValueError("Model not trained yet.")

        words_generated = self.estimator.generate(nwords, text_seed, random_seed)
        if detokenize:
            from nltk.tokenize.treebank import TreebankWordDetokenizer
            detokenizer = TreebankWordDetokenizer().detokenize
            detokenized = []
            for token in words_generated:
                if token == '<s>': continue
                if token == '</s>': break
                detokenized.append(token)
            return detokenizer(detokenized)
        else: return words_generated