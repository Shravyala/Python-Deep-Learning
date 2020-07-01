from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)
predicted = clf.predict(X_test_tfidf)
score = round(metrics.accuracy_score(twenty_test.target, predicted), 4)
print("MultinomialNB accuracy is: ", score)


# Changing the classifier to SVM
print("============= SVM ============")
from sklearn.svm import SVC
clf1 = SVC(kernel='linear')
clf1.fit(X_train_tfidf, twenty_train.target)
predicted1 = clf1.predict(X_test_tfidf)
score1 = round(metrics.accuracy_score(twenty_test.target, predicted1), 4)
print("SVM accuracy is:", score1)


# change the tfidf vectorizer to use bigram
tfidf_Vect2 = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf2 = tfidf_Vect2.fit_transform(twenty_train.data)
clf2 = MultinomialNB()
clf2.fit(X_train_tfidf2, twenty_train.target)
twenty_test2 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf2 = tfidf_Vect2.transform(twenty_test.data)
predicted2 = clf2.predict(X_test_tfidf2)
score2 = round(metrics.accuracy_score(twenty_test2.target, predicted2), 4)
print("MultinomialNB accuracy when using bigram is: ", score2)


# Set argument stop_words='english'
tfidf_Vect3 = TfidfVectorizer(stop_words='english')
X_train_tfidf3 = tfidf_Vect3.fit_transform(twenty_train.data)
clf3 = MultinomialNB()
clf3.fit(X_train_tfidf3, twenty_train.target)
twenty_test3 = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf3 = tfidf_Vect3.transform(twenty_test.data)
predicted3 = clf3.predict(X_test_tfidf3)
score3 = round(metrics.accuracy_score(twenty_test3.target, predicted3), 4)
print("MultinomialNB accuracy when adding the stop-words is: ", score3)

