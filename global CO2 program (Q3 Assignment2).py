# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 01:13:29 2020

@author: Sidra Shaikh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('global_co2.csv')

X = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1:2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 14)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Year')
plt.ylabel('Total')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Total')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Year')
plt.ylabel('Total')
plt.show()

# Predicting a new result with Linear Regression
print('2011(Linear Regression)')
print(lin_reg.predict([[2011]]))

# Predicting a new result with Polynomial Regression
print('2011(Polynomial Regression)')
print(lin_reg_2.predict(poly_reg.fit_transform([[2011]])))

# Predicting a new result with Linear Regression
print('2012(Linear Regression)')
print(lin_reg.predict([[2012]]))
# Predicting a new result with Polynomial Regression
print('2012(Polynomial Regression)')
print(lin_reg_2.predict(poly_reg.fit_transform([[2012]])))

# Predicting a new result with Linear Regression
print('2013(Linear Regression)')
print(lin_reg.predict([[2013]]))

# Predicting a new result with Polynomial Regression
print('2013(Polynomial Regression)')
print(lin_reg_2.predict(poly_reg.fit_transform([[2013]])))