#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:36:18 2019

@author: sreekanth.narayanan
"""


# Data preprocessing template 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set 

dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values # select everything except the last column 
y = dataset.iloc[:, 1].values #selct the last column of the data 

# SPlit the data set to training data set and test data set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0) # 1/3 in test and rest in the train 

# Fitting a simple linerar regression to our training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting the values based on the regression model 
y_pred = regressor.predict(X_test)

#visualizing the results and the predictions 
plt.scatter(X_train, y_train, color = 'red')
plt.plot (X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (training set)')
plt.xlabel ('Years of Exp')
plt.ylabel ('Salary')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot (X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel ('Years of Exp')
plt.ylabel ('Salary')
plt.show()
