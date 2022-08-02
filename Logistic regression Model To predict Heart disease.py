# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 17:13:22 2022

@author: Asim Pardeshi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#IMPORTING THE DATASET
dataset=pd.read_csv(r'C:\Users\Asim Pardeshi\OneDrive\Desktop\Datascience\framingham.csv')

#split dataset in X ,Y 
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#IMPUTING THE MISSNG VALUES
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(x)
x=imputer.transform(x)

#SPLITTING DATASET INTO TRAINING AND TESTING PHASE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#FEATURE SCALLING
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

# Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)

#PREDICTING THE TEST RESULTS
y_pred=classifier.predict(x_test)

#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#ACCURACY OF MODEL
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias=classifier.score(x_train,y_train)
bias

variance=classifier.score(x_test,y_test)
variance
