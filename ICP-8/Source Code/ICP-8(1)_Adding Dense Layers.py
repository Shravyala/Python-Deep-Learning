# importing libraries
import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Read Dataset
dataset = pd.read_csv("diabetes.csv", header=None).values

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,0:8], dataset[:,8],test_size=0.25, random_state=87)

# Apply Model
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=8, activation='relu')) # hidden layer

# Add another layer that have 10 neurals
my_first_nn.add(Dense(10, activation='relu')) # hidden layer
my_first_nn.add(Dense(50, activation='relu'))  # hidden layer

# last layer with 1 neural
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
# Configure the learning process
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, initial_epoch=0)
print(my_first_nn.summary())
# Evaluate Model
print(my_first_nn.evaluate(X_test, Y_test))