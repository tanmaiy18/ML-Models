#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

titanic=sns.load_dataset('titanic')
titanic.head()



# %%
#No. of rows and columns in the dataset
titanic.shape


# %%
#Some Statistics
titanic.describe()


# %%
#No. of survivors count
titanic["survived"].value_counts()


# %%
sns.countplot(titanic['survived'])


# %%
#Cols Subplot 
cols=['who','sex','pclass','sibsp','parch','embarked']
n_rows=3
n_cols=2
fig,axs=plt.subplots(n_rows,n_cols,figsize=(n_cols*3.2,n_rows*3.2))

for r in range(0,n_rows):
    for c in range(0,n_cols):
        i=r*n_cols+c
        ax=axs[r][c]
        sns.countplot(titanic[cols[i]],hue=titanic['survived'],ax=ax)
        ax.set_title(cols[i])
        ax.legend(title="survived",loc="upper right")
plt.tight_layout()


# %%
titanic.groupby('sex')[['survived']].mean()


# %%
titanic.pivot_table('survived',index='sex',columns='class')


# %%
titanic.pivot_table('survived',index='sex',columns='class').plot()


# %%
sns.barplot(x='class',y='survived',data=titanic)


# %%
#Age Group (<15 and >15)
age=pd.cut(titanic['age'],[0,15,80])
titanic.pivot_table('survived',['sex',age],'class')


# %%
#Age Group (<20 and >20)
age=pd.cut(titanic['age'],[0,20,80])
titanic.pivot_table('survived',['sex',age],'class')


# %%
plt.scatter(titanic['fare'],titanic['class'],color='purple',label='Passenger Paid')
plt.ylabel('Class')
plt.xlabel('Price/fare')
plt.title('Price of Each class')
plt.legend()
plt.show()


# %%
#All the empty no. of instances or fields in the columns
titanic.isna().sum()


# %%
#To see the count of each values involved in the column fields in the dataset
for val in titanic:
    print(titanic[val].value_counts())
    print()


# %%
#To remove redunant data
titanic=titanic.drop(['deck','embark_town','alive','class','who','alone','adult_male'],axis=1)
titanic=titanic.dropna(subset=['embarked','age'])


# %%
titanic.shape


# %%
#This is giving us the datatypes of the dataset
titanic.dtypes


# %%
#To know Unique Value of Sex and Embarked
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


# %%
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
#%%
labelencoder.fit(titanic["sex"])

titanic["sex"]=labelencoder.transform(titanic['sex'])

#%%
labelencoder.fit(titanic["embarked"])

titanic["embarked"]=labelencoder.transform(titanic['embarked'])

# %%
print(titanic['sex'].unique())
print(titanic['embarked'].unique())


# %%
titanic.dtypes


# %%
#Splitting data into dependent 'x' and 'y' variables
x=titanic.iloc[:,1:8].values
y=titanic.iloc[:,0].values


# %%
train, test = train_test_split(titanic, test_size = 0.2, random_state = 1)
def x_and_y(frame):
    inden = frame.drop(["survived"], axis = 1)
    target = frame["survived"]
    return inden, target

x_train, y_train = x_and_y(train)
x_test, y_test = x_and_y(test)

#%%
#Splitting the data into 80% training set and 20% testing set
from sklearn.model_selection import train_test_split
# x_train, y_train, x_test, y_test
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=12)


# %%
#Scale the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test=sc.transform(x_test)


# %%
#Create a function 
def models(x_train,y_train):

    #Logistic Regression
    from sklearn.linear_model import LogisticRegression
    log=LogisticRegression(random_state=0)
    log.fit(x_train,y_train)

    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
    knn.fit(x_train,y_train)

    #SVC (linear kernel)
    from sklearn.svm import SVC
    svc_lin=SVC(kernel='linear',random_state=0)
    svc_lin.fit(x_train,y_train)

    #SVC(RBF kernel)
    from sklearn.svm import SVC
    svc_rbf=SVC(kernel='rbf',random_state=0)
    svc_rbf.fit(x_train,y_train)

    #GaussianNB
    from sklearn.naive_bayes import GaussianNB
    gauss=GaussianNB()
    gauss.fit(x_train,y_train)

    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)
    tree.fit(x_train,y_train)

    #RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
    forest.fit(x_train,y_train)

    #Print Training Accuracy
    print('[0]Logistic Regression Training Accuracy: ',log.score(x_train,y_train))
    print('[1]K Neighbours Training Accuracy: ',knn.score(x_train,y_train))
    print('[2]SVC Linear Training Accuracy: ',svc_lin.score(x_train,y_train))
    print('[3]SVC RBF Training Accuracy: ',svc_rbf.score(x_train,y_train))
    print('[4]Gaussian Training Accuracy: ',gauss.score(x_train,y_train))
    print('[5]Decission Tree Training Accuracy: ',tree.score(x_train,y_train))
    print('[6]Random Forest Training Accuracy: ',forest.score(x_train,y_train))


    return log,knn,svc_lin,svc_rbf,gauss,tree,forest


# %%
#Get and Train all Models

model=models(x_train,y_train)
print(y_train)

# %%
#Show the confusion matrix and accuracy for all models on test data
from sklearn.metrics import confusion_matrix

for i in range(len(model)):
     cm=confusion_matrix(y_test,model[i].predict(x_test))

     #Extract TrueNegative,TruePsitive,FalseNegative,FalsePositive
     TN,FP,FN,TP=confusion_matrix(y_test,model[i].predict(x_test)).ravel()

     test_score=(TP+TN)/(TP+TN+FP+FN)

     print(cm)
     print('Model[{}] Testing Accuracy="{}"'.format(i,test_score))
     print()

# %% [markdown]
# Model 6 that is Random Forest Classifier did great

# %%
#Features importance
forest=model[6]
importances=pd.DataFrame({'feature':titanic.iloc[:,1:8].columns,'importance':np.round(forest.feature_importances_,3)})
importances=importances.sort_values('importance',ascending=False).set_index('feature')
importances


# %%
#Visualise the importances
importances.plot.bar()


# %%
pred=model[6].predict(x_test)
print(pred)

print()

print(y_test)#the actual values


# %%
survival_test=[[1,0,38,1,2,20,2]]

#Scaling
from sklearn.preprocessing import StandardScaler

survival_scaled=sc.transform(survival_test)


#%%
#Using RandomForest to predict
pred= forest.predict(survival_scaled)
print(pred)

#%%
if pred==0:
    print("OOPS! You Didn't Make It Alive RIP")
else:
    print("Congrats! You Survived ,Tell Us The Story")
# %%

# %%