from nltk.util import pad_sequence
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.util import everygrams
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import flatten

# get text
import io
import os
import requests

if os.path.isfile('language-never-random.txt'):
    with io.open('language-never-random.txt', encoding='utf8') as fin:
        text = fin.read()
else:
    url = "https://gist.githubusercontent.com/alvations/53b01e4076573fea47c6057120bb017a/raw/b01ff96a5f76848450e648f35da6497ca9454e4a/language-never-random.txt"
    text = requests.get(url).content.decode('utf8')
    with io.open('language-never-random.txt', 'w', encoding='utf8') as fout:
        fout.write(text)

from ngram import NGrams
model = NGrams(text, 3)
model.train()
print("counting of word the: ",model.count('the'))
print("counting of word never|language is: ",model.count('never', 'language', 'is'))
print("probability of word the: ",model.probability('the'))
print("probability of word never|language is: ",model.probability('never', 'language', 'is'))
print("first 10 words: ",model.generate(20))