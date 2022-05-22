#!/usr/bin/env python
# coding: utf-8

# In[1]:


import datetime as dt

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor


# In[26]:


df=pd.read_csv('glass.csv',index_col=[0])
df.head()


# In[27]:


df.shape


# In[28]:


df.info()


# In[29]:


df.isnull().sum()


# In[30]:


df.describe()


# In[31]:


df.dtypes


# In[32]:


df.corr()


# In[33]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True,cmap='viridis',linewidths=0.5)


# In[34]:


sns.pairplot(data=df)


# In[35]:


X = df.drop('Type of glass',axis=1)
X = np.cbrt(X) 


# In[36]:


Y = df['Type of glass']


# In[37]:


Y.value_counts()


# In[38]:


Y = Y.map({1:0,2:1,3:2,5:3,6:4,7:5})


# In[72]:


from sklearn.neighbors import KNeighborsClassifier
en_yakin = KNeighborsClassifier(n_neighbors = 6)
en_yakin.fit(X,Y)
print(en_yakin)
KNeighborsClassifier(algorithm = 'auto', leaf_size = 30 , metric = 'minkowski', metric_params= None, n_jobs = 1 , n_neighbors = 6 , p =2, weights='uniform')


# In[84]:


prediction = en_yakin.predict(X)
print(prediction)      

print(np.array(Y))


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2, random_state = 123, stratify=Y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[77]:


print(X_train.head())
print(y_train.head())


# In[78]:


from sklearn.neighbors import KNeighborsClassifier
en_yakin = KNeighborsClassifier(n_neighbors = 6)
en_yakin.fit(X_train,y_train)

KNeighborsClassifier(algorithm = 'auto', leaf_size = 30 , metric = 'minkowski', metric_params= None, n_jobs = 1 , n_neighbors = 6 , p =2, weights='uniform')


# In[85]:


prediction = en_yakin.predict(X_test)
prediction


# In[80]:


np.array(y_test)


# In[81]:


en_yakin.score(X_test, y_test)


# In[86]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))


# In[ ]:




