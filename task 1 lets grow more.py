#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading iris dataset

# In[4]:


data = pd.read_csv("iris.data", header=None) 
data.head() #first five rows of the dataset


# In[5]:


data.tail() #last five rows of the dataset


# In[6]:


data_headers = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
data.to_csv("Iris.csv", header=data_headers, index=False)


# In[7]:


ndata = pd.read_csv("Iris.csv")
ndata.head()


# # Getting Data info

# In[8]:


ndata.info()


# In[9]:


ndata.describe()


# In[10]:


ndata.isnull().sum() #checking for null data


# In[11]:


ndata.groupby('Species').size() #checking size of each species


# # Plotting

# In[12]:


sns.pairplot(data = ndata, hue='Species')
plt.show()


# # Comparison of each species with the features

# In[20]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.barplot(x = 'Species', y = 'SepalLength', data = ndata)
plt.subplot(2,2,2)
sns.barplot(x = 'Species', y = 'SepalWidth', data = ndata)
plt.subplot(2,2,3)
sns.barplot(x = 'Species', y = 'PetalLength', data = ndata)
plt.subplot(2,2,4)
sns.barplot(x = 'Species', y = 'PetalWidth', data = ndata)


# # Defining the dependent and independent variable
# 

# In[21]:


X = ndata.drop(columns="Species")
y = ndata["Species"]


# # Splitting into training and testing dataset

# In[26]:


pip install scikit-learn


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)


# # Building the Model
# Using the Random Forest Classifier Algorithm

# In[29]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


# # Making the confusion Matrix

# In[30]:


from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Conclusion
# Hence it can be concluded that Iris Flower Classification with Random Forest Model has an accuracy of approximately 96%

# In[ ]:




