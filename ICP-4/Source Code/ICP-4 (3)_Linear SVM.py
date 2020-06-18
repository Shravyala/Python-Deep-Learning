from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

glass = pd.read_csv('glass.csv')
x = glass[['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']]
y = glass['Type']

# Use cross validation to create training and testing part
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print("Shape of Training data")
print(X_train.shape, y_train.shape)
print("*"*50)
print("Shape of Testing data")
print(X_test.shape, y_test.shape)

svm = SVC()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print('Accuracy of SVM classifier on training set:', round(svm.score(X_train, y_train)*100,2))
print('Accuracy of SVM classifier on test set:', round(svm.score(X_test, y_test)*100,2))
print("Classification Report:\n",classification_report(y_test,y_pred))