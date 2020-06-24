# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Reading the data
train = pd.read_csv('winequality-red.csv')

# Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()

# Printing the top 3 correlated features to the target label(quality)
print("Top 3 correlated features are shown below")
print (corr['quality'].sort_values(ascending=False)[1:4], '\n')

# Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False))
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

# Handling missing value or Deleting the null values
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print()
print('Now, total nulls in data is: ', sum(data.isnull().sum() != 0))

# Build a linear model
y = np.log(train.quality)
X = data.drop(['quality'], axis=1)
# Split data into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
# Create the model
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)

# Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

# visualize

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,
            color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()

