import nltk
from nltk.util import ngrams


sentence = open('input.txt', encoding="utf8").read()
print(sentence)

# (a) Tokenization
print("\n============== Tokenization ==============\n")
from nltk import word_tokenize
from nltk import sent_tokenize
wtokens = nltk.word_tokenize(sentence)
stokens = nltk.sent_tokenize(sentence)
print(wtokens)
print(stokens)






