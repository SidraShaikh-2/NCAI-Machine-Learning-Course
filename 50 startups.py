# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 01:13:29 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :1].values # we set 0 as newyork and 1 as california
y = dataset.iloc[:, 1:2].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Linear regression (Training set)')
plt.xlabel('Newyork=0, California=1')
plt.ylabel('Profit')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('With linear regression')
plt.xlabel('Newyork=0, California=1')
plt.ylabel('Profit')
plt.show()

"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Newyork=0, California=1')
plt.ylabel('Profit')
plt.show()
print('Profit of Neyork')
print(regressor.predict([[0]]))
print('Profit of California')
print(regressor.predict([[1]]))