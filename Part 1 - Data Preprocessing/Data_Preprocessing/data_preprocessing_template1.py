#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:32:16 2019

@author: sreekanth.narayanan
"""

# Data preprocessing template 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the data set 

dataset = pd.read_csv('Data.csv')
X= dataset.iloc[:,:-1].values # select everything except the last column 
y = dataset.iloc[:, 3].values #selct the last column of the data 

# SPlit the data set to training data set and test data set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # 20% in test and 80% in the train 

# Feature scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler ()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""



