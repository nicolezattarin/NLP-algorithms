import re
from collections import Counter
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    """
    We provide an example of how to use the preprocess class and the skipgram class.
    Given a text file, we preprocess the text and train a skipgram model.
    Finally, we visualize the embeddings and the loss history.
    """

    # import the model
    from skipgram import SkipGram

    # load the corpus from the text file
    data = 'test.txt'
    with open(data) as f:
        text = f.read()

    # create the model and train it
    model = SkipGram(text, dim_embed=200, lr=0.001)
    model.train(n_epochs=200, batch_size=500, n_samples=5, verbose=True)

    # gets the history of the loss during training
    history = model.history_loss()

    # visualize the embeddings and the loss history
    # only the vectors:
    embeddings = model.embeddings()
    print("FIRST EMBEDDING OF THE VOCABULARY:\n", embeddings[:1])
    # dictonary of words and their embeddings:
    the_embed = model.embedding('the')
    print("EMBEDDING OF THE WORD 'THE':\n", the_embed)

    # visualize the embeddings in a scatter plot and save it to a file
    model.visualize_embed(100, figsize=(13,13), s=20)

    # plot the history of the loss to a file
    model.plot_loss(color='r', linewidth=2)

    # save the model to a file
    model.save_model("skipgram_model.pt")



if __name__ == '__main__':
    main()