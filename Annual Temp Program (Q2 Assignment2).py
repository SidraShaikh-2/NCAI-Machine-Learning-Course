# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 01:13:29 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('annual_temp.csv')

X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict([[2017]])

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Year')
plt.ylabel('Mean')
plt.show()
print('Prediction of 2016')
print(regressor.predict([[2016]]))
print('prediction of 2017')
print(regressor.predict([[2017]]))