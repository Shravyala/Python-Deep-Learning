# importing the pandas software library for data manipulation and analysis
import pandas as pd
# importing the numPy package
import numpy as np
# importing the matplotlib.pyplot for plotting the graph
import matplotlib.pyplot as plt

# Set up the output screen
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = [10, 6]
# Reading the data from file train1.csv
train1 = pd.read_csv('train1.csv')

# Display the scatter plot of GarageArea and SalePrice
plt.scatter(train1.GarageArea, train1.SalePrice, color='red')
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()

# calculate interquartile range
print(train1.GarageArea.describe())

# Delete the outlier value of GarageArea
outlier_drop = train1[(train1.GarageArea > 334) & (train1.GarageArea < 576)]

# Display the scatter plot of GarageArea and SalePrice after deleting
plt.scatter(outlier_drop.GarageArea, outlier_drop.SalePrice, color='green')
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.show()
