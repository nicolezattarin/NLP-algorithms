import re
from collections import Counter
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    with open('test.txt') as f:
        text = f.read()

    """test preprocesing """

    from preprocess import Preprocess
    preprocess = Preprocess()
    words = preprocess.cleaning(text)
    # print(words)
    print("Total words in text: {}".format(len(words)))
    print("Unique words: {}".format(len(set(words))))
    
    # print("\n\n")
    # create_lookup_tables
    int_words = preprocess._text_to_int(words)
    # print("int words ", int_words)

    print ("\n\n")
    #subsampling
    int_words = preprocess.subsampling(words)
    # print("int words ", int_words)


    """ test train"""
    from skipgram import SkipGram
    # read in the extracted text file      
    with open('test.txt') as f:
        text = f.read()

    sg = SkipGram(text, dim_embed=300)
    sg.train(n_epochs=10, batch_size=512, n_samples=5)

    history = sg.history_loss()
    print(history[:10])
    embeddings = sg.embeddings()
    print(embeddings[:10])

    sg.visualize_embed(100, figsize=(13,13), s=20)

    sg.save_model("skipgram_model.pt")

    print(sg.embedding(2))
    print(sg.embedding('and'))
    sg.plot_loss(color='r', linewidth=2)

if __name__ == '__main__':
    main()