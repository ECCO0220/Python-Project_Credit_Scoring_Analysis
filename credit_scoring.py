#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[25]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression


# In[5]:


dataset = pd.read_csv(r'/Users/jin.zhao/Desktop/Scorecard.csv')


# In[6]:


# count of rows and columns
dataset.shape


# In[7]:


# count of rows and columns
dataset.shape


# In[8]:


#shows first few rows of the code
dataset.head(5)


# In[9]:


dataset.describe()


# In[10]:


dataset.info()


# In[11]:


#1.dropping customer ID column 
dataset=dataset.drop('ID',axis=1)
dataset.shape


# In[12]:


#2. explore missing values
dataset.isna().sum()


# In[16]:


sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[17]:


# filling missing values with mean
dataset=dataset.fillna(dataset.mean())


# In[18]:


#check-explore missing values post missing value fix
dataset.isna().sum()


# In[19]:


# 1. count of good loans (0) and bad loans (1)
dataset['TARGET'].value_counts()


# In[20]:


# show in countplot chart
# set style
sns.set_style('white')
# change size
plt.figure(figsize=(2,6))
# countplot chart
sns.countplot(x='TARGET',data=dataset,palette='coolwarm')
# Spine Removal
sns.despine(left=True)


# In[21]:


# 2. data summary across 0 & 1
dataset.groupby('TARGET').mean()


# In[22]:


# Can do entire dataframe with orient='h'
sns.boxplot(data=dataset,palette='rainbow',orient='h')


# In[23]:


y = dataset.iloc[:, 0].values
X = dataset.iloc[:, 1:29].values


# In[27]:


# splitting dataset into training and test (in ratio 80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[28]:


# Data Normalization-scale all independent variables between 0 and 1.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[29]:


# Exporting Normalisation Coefficients for later use in prediction
import joblib
joblib.dump(sc, r'SCOREDCARD_NORMALISATION')


# In[30]:


# Train and fit a logistic regression model on the training set.
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


# In[31]:


# Exporting Logistic Regression Classifier model for later use in prediction
import joblib
joblib.dump(classifier, r'CLASSIFIER_SCORECARD')


# In[32]:


# generate probabilities 
predictions = classifier.predict_proba(X_test)
predictions


# In[33]:


print(confusion_matrix(y_test,y_pred))


# In[34]:


print(accuracy_score(y_test, y_pred))


# In[35]:


# writing model output file
df_prediction_prob = pd.DataFrame(predictions, columns = ['prob_0', 'prob_1'])
df_prediction_target = pd.DataFrame(classifier.predict(X_test), columns = ['predicted_TARGET'])
df_test_dataset = pd.DataFrame(y_test,columns= ['Actual Outcome'])
dfx=pd.concat([df_test_dataset, df_prediction_prob, df_prediction_target], axis=1)
dfx.to_csv(r"SCOREDCARD_MODEL_PREDICTION.csv", sep=',', encoding='UTF-8')
dfx.head()


# In[ ]:




