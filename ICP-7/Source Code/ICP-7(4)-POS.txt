import nltk
from nltk.util import ngrams


sentence = open('input.txt', encoding="utf8").read()
print(sentence)

wtokens = nltk.word_tokenize(sentence)

#POS
print("\n============== POS ==============\n")
print("Parts of Speech")
POS = nltk.pos_tag(wtokens)
print(POS)