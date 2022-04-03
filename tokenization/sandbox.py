import numpy as np
import pandas as pd
from text_procesing import load_data, nltk_tokenize, token_spans


def main():
    path_ita = "datasets/trial.txt" 
    path_eng = "datasets/10000_4.txt"

    # Load data from files
    print("\n\nENGLISH")
    print("Loading data...")
    w_tokens, w_type = load_data(path_eng)
    print(w_tokens, "\n")
    print(w_type)
    print("\n\nTokenizing...")
    text_eng = open(path_eng, 'r').read()
    tokens = nltk_tokenize(text_eng, tokenization_type='word', language='english')
    print(tokens)
    print("\n\nToken spans...")
    spans = token_spans(text_eng, language='english')
    print(spans)

    
    print("\n\nITALIAN")
    print("Loading data...")
    w_tokens, w_type = load_data(path_ita)
    print(w_tokens, "\n")
    print(w_type)
    print("\n\nTokenizing...")
    text_ita = open(path_ita, 'r').read()
    tokens = nltk_tokenize(text_ita, tokenization_type='word', language='italian')
    print(tokens)
    print("\n\nToken spans...")
    spans = token_spans(text_ita, language='italian')
    print(spans)

    print ("\n\n")
    print ("BPETokenizer")
    from text_procesing import BPE_tokenizer
    print(BPE_tokenizer(text_eng, 'datasets/bpe_roberta/vocab.json', 'datasets/bpe_roberta/merges.txt'))    



if __name__ == '__main__':
    
    main()