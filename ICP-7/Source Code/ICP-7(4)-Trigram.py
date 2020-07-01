import nltk
from nltk.util import ngrams
from nltk import trigrams

sentence = open('input.txt', encoding="utf8").read()
print(sentence)

wtokens = nltk.word_tokenize(sentence)
# Trigram
print("\n============== Trigram ==============\n")
print("TRIGRAMS:")
trig = trigrams(wtokens)
for x in trig:
  print(x)