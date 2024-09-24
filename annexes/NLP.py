
import nltk # https://www.nltk.org/install.html
import numpy # https://www.scipy.org/install.html
import matplotlib.pyplot # https://matplotlib.org/downloads.html
import tweepy # https://github.com/tweepy/tweepy
import TwitterSearch # https://github.com/ckoepp/TwitterSearch
import unidecode # https://pypi.python.org/pypi/Unidecode
import langdetect # https://pypi.python.org/pypi/langdetect
import langid # https://github.com/saffsd/langid.py
import gensim # https://radimrehurek.com/gensim/install.html
# nltk.download()

## Text Analysis Using nltk.text

from nltk.tokenize import word_tokenize
from nltk.text import Text
my_string = "Two plus two is four, minus one that's three — quick maths. Every day man's on the block. Smoke trees. See your girl in the park, that girl is an uckers. When the thing went quack quack quack, your men were ducking! Hold tight Asznee, my brother. He's got a pumpy. Hold tight my man, my guy. He's got a frisbee. I trap, trap, trap on the phone. Moving that cornflakes, rice crispies. Hold tight my girl Whitney."
tokens = word_tokenize(my_string)
tokens = [word.lower() for word in tokens]
# tokens[:5]
t = Text(tokens)
# t
f = open('my-file.txt','rU') # Opening a file with the mode 'U' or 'rU' will open a file for reading in universal newline mode. All three line ending conventions will be translated to a "\n"
raw = f.read()
t.concordance('uckers') # concordance() is a method of the Text class of NLTK. It finds words and displays a context window. Word matching is not case-sensitive.
# concordance() is defined as follows: concordance(self, word, width=79, lines=25). Note default values for optional params.

# Displaying 1 of 1 matches:
# girl in the park , that girl is an uckers . when the thing went quack quack q
t.collocations() # def collocations(self, num=20, window_size=2). num is the max no. of collocations to print.

t.count('quack')

t.index('two')
t.similar('brother') # similar(self, word, num=20). Distributional similarity: find other words which appear in the same contexts as the specified word; list most similar words first.
t.dispersion_plot(['man', 'thing', 'quack']) # Reveals patterns in word positions. Each stripe represents an instance of a word, and each row represents the entire text.
t.plot(20) # plots 20 most common tokens
t.vocab()
