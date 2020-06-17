import numpy as np    # imports the package
Z = np.random.uniform(1, 20, 20)   # defining the array of 20 float values ranging from 1 to 20
print(Z)
x = Z.reshape((4,5))    # reshaping the array to 4 rows and 5 columns
print(x)
y = np.where(x == np.amax(x, axis=1, keepdims=True), 0, x)  # replacing the large value in the array to 0
print("After replacing  the max in each row by 0")
print(y)