#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:32:16 2019

@author: sreekanth.narayanan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set 

dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values # select everything except the last column 
y = dataset.iloc[:, 3].values #selct the last column of the data 

#take the mean and replace the missing data 
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #0 is mean of the column 
imputer = imputer.fit(X[:, 1:3]) # indexes 1 and 2 - upper bound is excluded 
X[:, 1:3] = imputer.transform(X[:, 1:3]) # apply the transformation 

# Encode our categorical data 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0]) # fit to the first column of the data and encode to numbers

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# SPlit the data set to training data set and test data set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # 20% in test and 80% in the train 

# Feature scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler ()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

