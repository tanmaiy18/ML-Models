#%%
'''
* @FileName : IPL_Model.py
* @Author : Tanmaiy Reddy Arikatla
* @Brief : To predict the total runs scored by an IPL Team.
* @Date : 17 FEB 2021
*
* Copyright (C) 2021
'''
#%%
# Importing Libraries 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

#%%
#Importing Dataset

df = pd.read_csv("./ipl2017.csv")
df.head()
# %%
#Checking For Null Values

df.isna().sum()

#%%
#Dividing Target and Features 
y=df["total"]
x=df.drop(["total","date"],axis=1)
temp=x

#%%
#Viewing Bat_team and total graph
plt.scatter(y,x["bat_team"])
plt.show()

#%%
#Viewing Bowl_team and total graph
plt.scatter(y,x["bowl_team"])
plt.show()

#%%
#One Hot Encoding String Datasets

x=pd.get_dummies(x,["venue","bat_team","bowl_team","batsman","bowler"])

#%% 
# Splitting Training And Testing Dataset

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=28)

#%% 
# Fitting and scaling the data 

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
#Training Our Model on training data

model = RandomForestRegressor(oob_score=True)
model.fit(X_train,y_train)
#%%
# Testing Our Model On testing data 

model.score(X_test,y_test)
#%%
#Testing Our Model On OOB Data

model.oob_score_
#%%
#Predicting New Data

data = pd.DataFrame({"mid":[1],"venue":["Punjab Cricket Association Stadium, Mohali"],"bat_team":["Kings XI Punjab"],"bowl_team":["Royal Challengers Bangalore"],"batsman":["JR Hopes"],"bowler":["P Kumar"],"runs":[38],"wickets":[0],"overs":[1.2],"runs_last_5":[10],"wickets_last_5":[0],"striker":[12],"non-striker":[2]})
data = temp.append(data)

#Cleaning the data 
data=pd.get_dummies(data,["venue","bat_team","bowl_team","batsman","bowler"])

#selecting the orignal prediicting data
data = data.iloc[-1:]

#Scaling the data
data = scaler.transform(data)

#Making the prediction 
model.predict(data)

# %%
