#!/usr/bin/env python
# coding: utf-8

# ### 1. Data Exploration

# In[515]:


import numpy as np
import pandas as pd 
df =pd.read_csv("Titanic_train.csv")
df


# In[517]:


df.info()


# In[519]:


numerical_col =df.select_dtypes(include =['int64','float64']).columns
numerical_col



# In[521]:


categorical_col =df.select_dtypes(include =["object"]).columns
categorical_col


# In[523]:


#summary statistics---> Mean ,Median, Mode,Standard deviation,Minimum,Maximum
a1 =df["Age"].mean()
a2 =df["Age"].median()
a3 =df["Age"].mode()
a4 =df["Age"].std()
a5 =df["Age"].min()
a6 =df["Age"].max()
print("Mean of the age:", a1)
print("Median of the age:",a2)
print("Mode of the age:", a3)
print("Standard deviation of the age:", a4)
print("Minimum value of the age:", a5)
print("Maximum value of the age:", a6)


# In[525]:


a1 =df["SibSp"].mean()
a2 =df["SibSp"].median()
a3 =df["SibSp"].mode()
a4 =df["SibSp"].std()
a5 =df["SibSp"].min()
a6 =df["SibSp"].max()
print("Mean of the SibSp:", a1)
print("Median of the SibSp:",a2)
print("Mode of the SibSp:", a3)
print("Standard deviation of the SibSp:", a4)
print("Minimum value of the SibSp:", a5)
print("Maximum value of the SibSp:", a6)


# In[527]:


a1 =df["Parch"].mean()
a2 =df["Parch"].median()
a3 =df["Parch"].mode()
a4 =df["Parch"].std()
a5 =df["Parch"].min()
a6 =df["Parch"].max()
print("Mean of the Parch:", a1)
print("Median of the Parch:",a2)
print("Mode of the Parch:", a3)
print("Standard deviation of the Parch:", a4)
print("Minimum value of the Parch:", a5)
print("Maximum value of the Parch:", a6)


# In[529]:


a1 =df["Fare"].mean()
a2 =df["Fare"].median()
a3 =df["Fare"].mode()
a4 =df["Fare"].std()
a5 =df["Fare"].min()
a6 =df["Fare"].max()
print("Mean of the Fare:", a1)
print("Median of the Fare:",a2)
print("Mode of the Fare:", a3)
print("Standard deviation of the Fare:", a4)
print("Minimum value of the Fare:", a5)
print("Maximum value of the Fare:", a6)


# In[ ]:





# ### Histograms

# In[533]:


#EDA 
#histograms
import matplotlib.pyplot as plt 
print(df["Pclass"])
df["Pclass"].hist()


# In[534]:


print(df["Sex"])
df["Sex"].hist()


# In[536]:


print(df["Age"])
df["Age"].hist()


# In[537]:


print(df["SibSp"])
df["SibSp"].hist()


# In[539]:


print(df["Parch"])
df["Parch"].hist()


# In[540]:


print(df["Ticket"])
df["Ticket"].hist()


# In[541]:


print(df["Fare"])
df["Fare"].hist()


# In[543]:


print(df["Cabin"])
df["Cabin"].hist()


# In[544]:


print(df["Embarked"])
df["Embarked"].hist()


# ### Box plots 

# In[547]:


#box plots 
df.boxplot(column =["Pclass"], vert =False)


# In[549]:


#box plots 
df.boxplot(column =["Age"], vert =False)


# In[550]:


df.boxplot(column =["SibSp"], vert =False)


# In[551]:


df.boxplot(column =["Parch"], vert =False)


# In[552]:


df.boxplot(column =["Fare"], vert =False)


# In[553]:


import matplotlib.pyplot as plt 
plt.scatter(x =df["Pclass"], y =df["Survived"], color ='red')
plt.plot(df["Pclass"], df["Survived"],c ='black')
plt.show()


# In[554]:


import matplotlib.pyplot as plt 
plt.scatter(x =df["Sex"], y =df["Survived"], color ='blue')
plt.plot(df["Sex"], df["Survived"],c ='pink')
plt.show()


# In[555]:


import matplotlib.pyplot as plt 
plt.scatter(x =df["Age"], y =df["Survived"], color ='red')
plt.plot(df["Age"], df["Survived"],c ='yellow')
plt.show()


# In[556]:


import matplotlib.pyplot as plt 
plt.scatter(x =df["SibSp"], y =df["Survived"], color ='orange')
plt.plot(df["SibSp"], df["Survived"],c ='green')
plt.show()


# In[557]:


import matplotlib.pyplot as plt 
plt.scatter(x =df["Parch"], y =df["Survived"], color ='black')
plt.plot(df["Parch"], df["Survived"],c ='teal')
plt.show()


# In[558]:


import matplotlib.pyplot as plt 
plt.scatter(x =df["Ticket"], y =df["Survived"], color ='blue')
plt.plot(df["Ticket"], df["Survived"],c ='grey')
plt.show()


# In[559]:


import matplotlib.pyplot as plt 
plt.scatter(x =df["Fare"], y =df["Survived"], color ='red')
plt.plot(df["Fare"], df["Survived"],c ='purple')
plt.show()


# In[560]:


df.corr


# ### 2. Data Preprocessing

# In[562]:


print(df.isnull().sum())


# In[563]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Name"] =LE.fit_transform(df["Name"])
print(df["Name"])


# In[564]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Sex"] =LE.fit_transform(df["Sex"])
print(df["Sex"])


# In[565]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Ticket"] =LE.fit_transform(df["Ticket"])
print(df["Ticket"])


# In[566]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Cabin"] =LE.fit_transform(df["Cabin"])
print(df["Cabin"])


# In[567]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Embarked"] =LE.fit_transform(df["Embarked"])
print(df["Embarked"])


# In[568]:


'''
# Fill missing with median/mean
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing with mean/medain
df['Cabin'].fillna(df['Cabin'].median(), inplace=True)

# Fill missing with median/mean
df['Embarked'].fillna(df['Embarked'].median(), inplace=True)
'''


# In[570]:


df.corr()


# In[ ]:





# ### 3.Model Building

# In[406]:


#Data partition 
Y =df["Survived"]
X =df.iloc[:, 8:]
X.head()


# In[407]:


from sklearn.linear_model import LogisticRegression 
model =LogisticRegression()
model.fit(X, Y)


# In[408]:


df["Y_pred"] =model.predict(X)


# ### 4. Model Evaluation

# In[410]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
score =accuracy_score(Y, df["Y_pred"])
print("accuracy score:", np.round(score, 2))

ps=precision_score(Y, df["Y_pred"])
print("precision score:", np.round(ps, 2))

rs =recall_score(Y, df["Y_pred"])
print("recall score:", np.round(rs, 2))

f1s =f1_score(Y, df["Y_pred"])
print("F1 score:", np.round(f1s, 2))



# In[411]:


from sklearn.metrics import roc_auc_score
auc =roc_auc_score(Y, df["Y_pred"])
print("Area under curve:",  np.round(auc, 3))
print("Area under curve:", (auc * 100).round(3))


# In[412]:


from sklearn.metrics import roc_curve 

fpr,tpr,dummy =roc_curve(Y, df["Y_pred"])

import matplotlib.pyplot as plt 
plt.scatter(x =fpr, y =tpr)
plt.plot(fpr,tpr, color ="red")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


# ### 5. Interpretation

# In[414]:


model.intercept_


# In[415]:


df.corr()


# ### 6. Deployment with Streamlit:

# In[ ]:





# ### Test Data

# ### 1. Data Exploration

# In[419]:


import numpy as np
import pandas as pd 
df =pd.read_csv("Titanic_test.csv")
df


# In[420]:


df.info()


# In[421]:


numerical_col =df.select_dtypes(include =['int64','float64']).columns
numerical_col



# In[422]:


categorical_col =df.select_dtypes(include =["object"]).columns
categorical_col


# In[423]:


#summary statistics---> Mean ,Median, Mode,Standard deviation,Minimum,Maximum
a1 =df["Age"].mean()
a2 =df["Age"].median()
a3 =df["Age"].mode()
a4 =df["Age"].std()
a5 =df["Age"].min()
a6 =df["Age"].max()
print("Mean of the age:", a1)
print("Median of the age:",a2)
print("Mode of the age:", a3)
print("Standard deviation of the age:", a4)
print("Minimum value of the age:", a5)
print("Maximum value of the age:", a6)


# In[424]:


a1 =df["SibSp"].mean()
a2 =df["SibSp"].median()
a3 =df["SibSp"].mode()
a4 =df["SibSp"].std()
a5 =df["SibSp"].min()
a6 =df["SibSp"].max()
print("Mean of the SibSp:", a1)
print("Median of the SibSp:",a2)
print("Mode of the SibSp:", a3)
print("Standard deviation of the SibSp:", a4)
print("Minimum value of the SibSp:", a5)
print("Maximum value of the SibSp:", a6)


# In[425]:


a1 =df["Parch"].mean()
a2 =df["Parch"].median()
a3 =df["Parch"].mode()
a4 =df["Parch"].std()
a5 =df["Parch"].min()
a6 =df["Parch"].max()
print("Mean of the Parch:", a1)
print("Median of the Parch:",a2)
print("Mode of the Parch:", a3)
print("Standard deviation of the Parch:", a4)
print("Minimum value of the Parch:", a5)
print("Maximum value of the Parch:", a6)


# In[426]:


a1 =df["Fare"].mean()
a2 =df["Fare"].median()
a3 =df["Fare"].mode()
a4 =df["Fare"].std()
a5 =df["Fare"].min()
a6 =df["Fare"].max()
print("Mean of the Fare:", a1)
print("Median of the Fare:",a2)
print("Mode of the Fare:", a3)
print("Standard deviation of the Fare:", a4)
print("Minimum value of the Fare:", a5)
print("Maximum value of the Fare:", a6)


# ### Histograms

# In[428]:


#EDA 
#histograms
import matplotlib.pyplot as plt 
print(df["Pclass"])
df["Pclass"].hist()


# In[429]:


print(df["Age"])
df["Age"].hist()


# In[430]:


print(df["SibSp"])
df["SibSp"].hist()


# In[431]:


print(df["Parch"])
df["Parch"].hist()


# In[432]:


print(df["Sex"])
df["Sex"].hist()


# In[433]:


print(df["Ticket"])
df["Ticket"].hist()


# In[434]:


print(df["Fare"])
df["Fare"].hist()


# In[435]:


print(df["Cabin"])
df["Cabin"].hist()


# In[436]:


print(df["Embarked"])
df["Embarked"].hist()


# ### Boxplot

# In[438]:


#box plots 
df.boxplot(column =["Pclass"], vert =False)


# In[439]:


#box plots 
df.boxplot(column =["Age"], vert =False)


# In[440]:


df.boxplot(column =["SibSp"], vert =False)


# In[441]:


df.boxplot(column =["Parch"], vert =False)


# In[442]:


df.boxplot(column =["Fare"], vert =False)


# In[443]:


df.corr


# ### 2. Data Preprocessing

# In[596]:


print(df.isnull().sum())


# In[598]:


#fill missing values with Median or Mean 
df["Age"].fillna(df["Age"].median(), inplace =True)
df["Fare"].fillna(df["Fare"].median(), inplace =True)
df["Cabin"].fillna(df["Cabin"].median(), inplace =True) 


# In[600]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Name"] =LE.fit_transform(df["Name"])
print(df["Name"])


# In[602]:


#Label Encoder
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df["Name"]     =LE.fit_transform(df["Name"])
df["Sex"]      =LE.fit_transform(df["Sex"])
df["Ticket"]   =LE.fit_transform(df["Ticket"])
df["Cabin"]    =LE.fit_transform(df["Cabin"])
df["Embarked"] =LE.fit_transform(df["Embarked"])

df.head()


# In[604]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Sex"] =LE.fit_transform(df["Sex"])
print(df["Sex"])


# In[606]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Ticket"] =LE.fit_transform(df["Ticket"])
print(df["Ticket"])


# In[608]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Cabin"] =LE.fit_transform(df["Cabin"])
print(df["Cabin"])


# In[610]:


#DATA TRANSFORMATION
from sklearn.preprocessing import LabelEncoder 
LE =LabelEncoder()
df["Embarked"] =LE.fit_transform(df["Embarked"])
print(df["Embarked"])


# In[612]:


df.corr()


# ### 3.Model Building

# In[615]:


#Data partition 
Y =df["Survived"]
X =df.iloc[:, 8:]
X


# In[617]:


from sklearn.linear_model import LogisticRegression 
model =LogisticRegression()
model.fit(X, Y)


# In[619]:


df["Y_pred"] =model.predict(X)


# ### 4. Model Evaluation:

# In[621]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score 
score =accuracy_score(Y, df["Y_pred"])
print("accuracy score:", np.round(score, 2))

ps=precision_score(Y, df["Y_pred"])
print("precision score:", np.round(ps, 2))

rs =recall_score(Y, df["Y_pred"])
print("recall score:", np.round(rs, 2))

f1s =f1_score(Y, df["Y_pred"])
print("F1 score:", np.round(f1s, 2))


# In[641]:


from sklearn.metrics import roc_auc_score
auc =roc_auc_score(Y, df["Y_pred"])
print("Area under curve:",  np.round(auc, 3))
print("Area under curve:", (auc * 200).round(3))


# In[643]:


from sklearn.metrics import roc_curve 

fpr,tpr,dummy =roc_curve(Y, df["Y_pred"])

import matplotlib.pyplot as plt 
plt.scatter(x =fpr, y =tpr)
plt.plot(fpr,tpr, color ="red")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()


# ### 5. Interpretation:

# In[627]:


model.intercept_


# In[629]:


df.corr()


# ### Interview Questions:

# In[ ]:


'''
1. What is the difference between precision and recall?

-->Sensitivity / Recall:

    From the overall Actual positiveshow much our model successfully predicted them as positives. 
    Percentage of positives that are successfully classified as positives. 
True positive Rate 
    TPR = TP/(TP + FN)
     If FN decreases Sensitivity increases, If FN increases, Sensitivity Decreases

-->Precision:

    from the model predicted positives what Percentage of people are real positives.
    The approach here is to find what percentage of the model’s positive (1’s) predictions are accurate. 
    Precision is calculated as the number of correct positive predictions (TP) divided by the total number of positive predictions 
    (TP) / (TP + FP)


2. What is cross-validation, and why is it important in binary classification?

Model is fitted on training or known data. One must do the cross-validation & model tuning before making 
any conclusions about the results. Cross-validation is done to issues like over fitting and model tuning is 
done to get the best model parameters which can give best required results. Once you have chosen the 
models, then you can perform model tuning and cross-validation for each of the chosen models. Cross
validation is like repeatedly checking the model performance on unknown dataset and thereby increasing 
the assurance of the model performance on any data set which will be fed into this model in future.

                        { OR }
Cross-validation is a statistical technique used to evaluate the performance of a machine learning model and ensure that it generalizes 
well to unseen data. It is especially important in binary classification (where there are only two class labels, e.g., "positive" and "negative") 
because it helps avoid overfitting and gives a more reliable estimate of model performance.

Cross-validation involves splitting the dataset into multiple subsets (called folds), training the model on some folds, 
and testing it on the remaining fold(s). This process is repeated several times, and the results are averaged.

Common types:
    K-Fold Cross-Validation: The dataset is divided into K equal parts. The model is trained on K-1 folds and tested on the remaining fold. 
    This process repeats K times with a different test fold each time.

    Stratified K-Fold Cross-Validation: Similar to K-Fold, but ensures each fold has the same proportion of class labels. 
    Very useful for binary classification with imbalanced data.

Why is Cross-Validation Important in Binary Classification?
    1.Reduces Overfitting Risk
    2.Gives Robust Performance Metrics
    3.Handles Class Imbalance
    4.Supports Model Selection
'''


# In[ ]:




