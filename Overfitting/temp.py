# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Project 2.1
# %% [markdown]
# ## Importing the necessary libraries and modules

# %%
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score as ras 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler as MMS
from sklearn.linear_model import LogisticRegression as LR 
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import train_test_split as tts 

# %% [markdown]
# ## Reading the Dataset 

# %%
data = pd.read_csv(r"C:/Users/Tanmaiy/Documents/IBM/Assignment/Overfitting/train.csv")
pd.set_option('display.max_columns', None)
df = data.drop("id", axis = 1)  #Dropping the ID Column. 
df.head()

# %% [markdown]
# ## Analyzng the Dataset

# %%
print("This dataset has only continuous values as the prevailing dataype is {}." .format(np.unique(df.dtypes)[0]))
print("This dataset has {} rows and {} columns. ".format(df.shape[0], df.shape[1]))
print("This dataset has {} missing values. ".format(df.isnull().sum().sum()))

# %% [markdown]
# ## Separating the Dependent and Independent Variables

# %%
x = df.drop("target", axis = 1)
y = df["target"]
x.shape,y.shape

# %% [markdown]
# ## Scaling the Data using MinMaxScaler

# %%
mms = MMS()
x_scaled = mms.fit_transform(x)
x = pd.DataFrame(x_scaled, columns = x.columns)
x.head()

# %% [markdown]
# ## Collecting Error Values to see the Trend 

# %%
i = 0
lr = LR()
test_error = []
train_error = []
while(i<10):
    x_train,x_test,y_train,y_test = tts(x, y)
    lr.fit(x_train, y_train)
# For Training Data:
    train_predict = lr.predict(x_train)
    train_roc_auc = ras(y_score = train_predict, y_true = y_train)
    train_error.append(1-train_roc_auc)

# For Test Data:
    test_predict = lr.predict(x_test)
    test_roc_auc = ras(y_score = test_predict, y_true = y_test)
    test_error.append(1-test_roc_auc)

    i = i+1

# %% [markdown]
# ### Plotting Train and Test ROC AUC Errors

# %%
x_ticks = np.arange(0, 11, 1)
plt.xticks(x_ticks)
plt.plot(train_error , color = "blue", label = "Train Error")
plt.plot(test_error , color = "red", label = "Test Error")
plt.title("Train Error vs Test Error")
plt.xlabel("Iterations")
plt.ylabel("ROC AUC Error")
plt.legend()

# %% [markdown]
# From this graph, we can say that the Model is overfitted as the model performs extremely well on the training data but not on the validation data. 
# %% [markdown]
# ## Parameter Tuning
# %% [markdown]
# ### Removing the Columns that don't contribute much to the final prediction

# %%
lr.coef_


# %%
plt.figure(figsize=(10, 10), dpi=120, facecolor='w', edgecolor='b')
x = range(len(x_train.columns))
c = lr.coef_.reshape(-1)
plt.bar( x, c )
plt.xlabel( "Variables")
plt.ylabel('Coefficients')
plt.title('Coefficient plot')


# %%
Coefficients = pd.DataFrame({"Variables": x_train.columns, "Coefficients": lr.coef_.reshape(-1)})

Coefficients.head()

# %% [markdown]
# ### Removing Columns with Coefficients less than 0.5

# %%
sign_value = Coefficients[Coefficients["Coefficients"]>0.4]


# %%
subsets = df[sign_value["Variables"]]


# %%
i = 0
lr = LR()
test_error = []
train_error = []
while(i<10):
    x_train,x_test,y_train,y_test = tts(subsets, y)
    lr.fit(x_train, y_train)
# For Training Data:
    train_predict = lr.predict(x_train)
    train_roc_auc = ras(y_score = train_predict, y_true = y_train)
    train_error.append(1-train_roc_auc)

# For Test Data:
    test_predict = lr.predict(x_test)
    test_roc_auc = ras(y_score = test_predict, y_true = y_test)
    test_error.append(1-test_roc_auc)

    i = i+1

x_ticks = np.arange(0, 11, 1)
plt.xticks(x_ticks)
plt.plot(train_error , color = "blue", label = "Train Error")
plt.plot(test_error , color = "red", label = "Test Error")
plt.title("Train Error vs Test Error")
plt.xlabel("Iterations")
plt.ylabel("ROC AUC Error")
plt.legend()


# %%
x_train,x_test,y_train,y_test = tts(subsets, y, random_state = 56)
lr = LR()

#For Training Data:

lr.fit(x_train, y_train)
train_predict = lr.predict(x_train)
train_ras = ras(train_predict, y_train)

#For Test Data:


test_predict = lr.predict(x_test)
test_ras = ras(test_predict, y_test)

train_ras, test_ras


# %%
plt.figure(figsize=(10, 10), dpi=120, facecolor='w', edgecolor='b')
x = range(len(x_train.columns))
c = lr.coef_.reshape(-1)
plt.bar( x, c )
plt.xlabel( "Variables")
plt.ylabel('Coefficients')
plt.title('Coefficient plot')

# %% [markdown]
# ## Confusion Matrix

# %%
cf_matrix = confusion_matrix(y_true = y_test, y_pred = test_predict)
cf_matrix
sns.heatmap(cf_matrix, annot=True, fmt="", cmap='Greens')


# %%
lr.coef_

# %% [markdown]
# ## Cross Validation

# %%
folds = StratifiedKFold(n_splits=10)
for train_index, test_index in folds.split(subsets,y):
    x_train, x_test, y_train, y_test = subsets.iloc[train_index], subsets.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    lr.fit(x_train, y_train)
    test_predict = lr.predict(x_test)
    test_roc_auc = ras(y_score = test_predict, y_true = y_test)


# %%
cvs(lr, subsets, y, scoring="roc_auc", cv=10)


# %%
lr.coef_
cf_matrix = confusion_matrix(y_true = y_test, y_pred = test_predict)
cf_matrix
sns.heatmap(cf_matrix, annot=True, fmt="", cmap='Greens')

# %% [markdown]
# ## Predicting the Values of the Test Data, Dataset the model hasn't seen yet

# %%
test = pd.read_csv(r"C:/Users/Tanmaiy/Documents/IBM/Assignment/Overfitting/test.csv")


# %%
test.head()


# %%
test_df = test[sign_value["Variables"]]
test_df.head()


# %%
main_test_pred = lr.predict(test_df)


# %%
test["target"] = main_test_pred

# %% [markdown]
# ## Creating and Saving the Final Dataset

# %%
final = test[["id", "target"]]


# %%
#final.to_csv(r"C:\Users\acer\Desktop\final.csv", index = False)


