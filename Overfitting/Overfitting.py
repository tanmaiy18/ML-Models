# %% Importing Modules
import pandas as pd
import numpy as np
import seaborn as sns
import time,sys,os,warnings
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score as f1s


# %% Importing Data
warnings.filterwarnings("ignore")

pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
train_df=pd.read_csv("C:/Users/Tanmaiy/Documents/IBM/Assignment/Overfitting/train.csv")
test_df=pd.read_csv("C:/Users/Tanmaiy/Documents/IBM/Assignment/Overfitting/test.csv")

# %%
train_df.describe()
test_df.describe()

# %%
train_df.isnull().sum()
train_df.isnull().sum().sum()
test_df.isnull().sum()
test_df.isnull().sum().sum()
# SOO it means no null data
#%%
train_df=train_df.drop(["id"],axis=1)
id=test_df["id"]
test_df=test_df.drop(["id"],axis=1)

# %%
sns.countplot(train_df["target"])

# %%
import scipy.stats as stats
#  depen = Categorical, inden = Continous ==> pointbiserial test
features = []
alpha = 0.07
for col in train_df.columns:
    target,pvalue = stats.pointbiserialr(train_df["target"],train_df[col])
    
    if pvalue>alpha:
        pass
    else:
        features.append(col)

# %%
train_df = train_df[features]
features.pop(0)
test_df = test_df[features]

# %%
train, test = train_test_split(train_df, test_size = 0.2, random_state = 1)
def x_and_y(frame):
    inden = frame.drop(["target"], axis = 1)
    target = frame["target"]
    return inden, target

x_train, y_train = x_and_y(train)
x_test , y_test = x_and_y(test)
# %%
#Training
model=LogisticRegression()
model.fit(x_train,y_train)

train_pred=model.predict(x_train)
test_pred=model.predict(x_test)

train_f1 = f1s(train_pred, y_train)
test_f1 = f1s(test_pred, y_test)

print("\nThe F1 Score for the Training Set is: {:.2f}%\n".format(train_f1*100))
print("\nThe F1 Score for the Testing Set is: {:.2f}%\n".format(test_f1*100))

score_matrix =confusion_matrix(y_test,test_pred)
print("Confusion Matrix for the tested data is :\n",score_matrix,"\n")

#%%
#Predicting For Test Data set
test_pred = model.predict(test_df)
result=dict
print("To get the result of the test data set please remove the '#' from the above for loop")
#for i in range(len(id)):
#    print(id[i],test_pred[i])

#%%

# %%
