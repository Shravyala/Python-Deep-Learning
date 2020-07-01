import nltk
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer

sentence = open('input.txt', encoding="utf8").read()
print(sentence)

#Lemmatization
print("\n============== Lemmatizing ==============\n")
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize(sentence))