import numpy as np
import pandas as pd
import nltk

def load_data(path):
    """
    Loads the data from the given path.
    returns:
        w_tokens: list of token words
        w_type: dictionary of tokens and their counts
    """
    from nltk.tokenize import word_tokenize
    w_tokens = open(path, 'r').read().split()
    w_type = {v: k for k, v in enumerate(w_tokens)}
    return w_tokens, w_type

def nltk_tokenize(text, tokenization_type='word', language='english'):
    """
    Tokenizes the given text.
    args:
        text: string of text
        tokenization_type: string of type of tokenization
                           "word" for word tokenization
                           "wordpunct" for word and punctuation tokenization
                           "sentence" for sentence tokenization
    returns:
        tokens: list of token words
    """
    nltk.download('punkt')

    from nltk.tokenize import word_tokenize, wordpunct_tokenize, sent_tokenize
    if tokenization_type == 'word': tokens = word_tokenize(text, language=language)
    elif tokenization_type == 'wordpunct': tokens = wordpunct_tokenize(text, language=language)
    elif tokenization_type == 'sentence': tokens = sent_tokenize(text, language=language)

    return tokens

def token_spans(text, language='english'):
    """
    Returns the spans of the tokens in the given text.
    args:
        text: string of text
    returns:
        spans: list of tuples of token spans
    """
    nltk.download('punkt')
    from nltk.tokenize import WhitespaceTokenizer
    return list(WhitespaceTokenizer().span_tokenize(text))

def BPE_tokenizer(text, vocab, merges, unk_token='<unk>'):
    """
    Tokenizes the given text.
    args:
        text: string of text
        vocab: path to vocab file
        merges: path to token merges
        unk_token: string of unknown token
    returns:
        tokens: list of token words
    """
    from tokenizers import Tokenizer, AddedToken
    from tokenizers.models import BPE, WordPiece
    tokenizer = Tokenizer(BPE.from_file(vocab=vocab, merges=merges, unk_token=unk_token))
    return tokenizer.encode(text).tokens