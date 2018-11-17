# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 16:01:27 2018

@author: Karan Desai
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as datetime

# Importing the dataset
dataset = pd.read_csv('data_ahn.csv')
#dataset.drop('Age_group',axis=1,inplace=True)

# DATA CLEANING
dataset = dataset[dataset['Age'].isnull()==False]

# Replacing null MS DRG code with 0 which indicates no procedure performed and setting description to NONE
#Assuming procedure NONE beacuse the costs indicate that nothing complex was performed on the patient
dataset['MS DRG'][dataset['MS DRG'].isnull()] = 0
dataset['MS DRG Description'][dataset['MS DRG Description'].isnull()] = 'NONE'  

final_dataset = dataset[['Age','Admit Date','Attending Physician','MS DRG Description','LOS','Charges']]
# Convert date to number of days since 1 jan 1970 as 
epoch_0=datetime.datetime(1970,1,1)
final_dataset['Admit Date']=(pd.to_datetime(final_dataset['Admit Date'])-epoch_0) / np.timedelta64(1,'D')

# One hot encoding for categorical inputs
test = pd.get_dummies(data=final_dataset,drop_first=True)

# input data
final_input = test.drop('Charges',axis=1)


X = final_input.iloc[:,:].values
y = test.iloc[:, 3].values

# Taking care of missing dates in the data using mean strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 1)
imputer = imputer.fit(X[:, 1])
X[:, 1] = imputer.transform(X[:, 1])



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(fit_intercept=False)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)


from sklearn.externals import joblib

joblib.dump(regressor, "linear_regression_model_for_charges.pkl",compress=1)




from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


import seaborn as sns
corr = final_dataset.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

