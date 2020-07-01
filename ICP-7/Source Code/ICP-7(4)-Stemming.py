import nltk
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer

sentence = open('input.txt', encoding="utf8").read()
print(sentence)


#Stemming
print("\n============== STEMMING ==============\n")
print(" Porter STEMMING:")
porterStemmer = PorterStemmer()
print(porterStemmer.stem(sentence))

print(" Lancaster STEMMING:")
lancasterStemmer = LancasterStemmer()
print(lancasterStemmer.stem(sentence))

print(" Snowball STEMMING:")
snowballStemmer = SnowballStemmer('english')
print(snowballStemmer.stem(sentence))