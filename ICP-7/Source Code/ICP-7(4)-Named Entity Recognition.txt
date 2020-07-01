import nltk
from nltk.util import ngrams
from nltk import ne_chunk

sentence = open('input.txt', encoding="utf8").read()
print(sentence)

wtokens = nltk.word_tokenize(sentence)
POS = nltk.pos_tag(wtokens)
# Named Entity Recognition
print("\n============== Named Entity Recognition ==============\n")
print("Named Entity Recognition")
ner = ne_chunk(POS)
print("\nNamed Entity Recognition :", ner)