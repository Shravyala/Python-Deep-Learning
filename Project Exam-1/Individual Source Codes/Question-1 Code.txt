import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

# Read file
data = pd.read_csv("creditcard.csv")

# Training and testing of the dataset
x_data = data.drop("Class", axis=1)
y_data = data.Class
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=0)

# Visual representation of the sample
count = data["Class"].value_counts()
sample = ["Real", "Fraud"]
plt.title("Transactions")
plt.pie(count, labels=sample, autopct='%0.4f%%')
plt.show()

# Classification using Naive Bayes
model = GaussianNB()
model.fit(x_data, y_data)
predict = model.predict(X_test)
accuracy = accuracy_score(Y_test, predict) * 100
print("Naive Bayes classification:")
print("Test data accuracy %: " + str(accuracy))
print("Confusion matrix: ")
print(confusion_matrix(Y_test, predict))

# Undersampling as a solution to the unbalanced dataset
model = KNeighborsClassifier()
model.fit(x_data, y_data)
predict = model.predict(X_test)
accuracy = accuracy_score(Y_test, predict) * 100
print("kNN classification:")
print("Test data accuracy %: " + str(accuracy))
print("Confusion matrix: ")
print(confusion_matrix(Y_test, predict))

