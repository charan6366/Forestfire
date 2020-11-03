#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv("Forest_fire.csv")


# In[3]:


data = np.array(data)


# In[4]:


X = data[1:, 1:-1]
y = data[1:, -1]
y = y.astype('int')
X = X.astype('int')


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

pickle.dump(log_reg,open('model.pkl','wb'))

