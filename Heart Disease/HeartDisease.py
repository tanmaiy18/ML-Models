#%%
'''
* @FileName : HeartDisease.py
* @Author : Tanmaiy Reddy Arikatla
* @Brief : To predict if a person is suffering with as heart disease or not.
* @Date : 1 DEC 2020
*
* Copyright (C) 2020
'''
#%% 
# Importing Libraries
 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

#%% 
# Importing Data set
 
df = pd.read_csv(".\heart.csv")
df.head()

#%% 
# Spliting Target and Features Data 

y=df["target"]
X=df.drop("target",axis=1)

#%%
#  Cheacking For Null values 

X.isna().sum()
y.isna().sum()

#%%
#  Datatypes of our data

X.dtypes
y.dtypes
X.describe()

#%% 
# Splitting Training And Testing Dataset

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=18,stratify=y)

#%% 
# Fitting and scaling the data 

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% 
# Training the model

model = KNeighborsClassifier()
model.fit(X_train,y_train)

#%% 
# Testing The model

model.score(X_test,y_test)
# %%
