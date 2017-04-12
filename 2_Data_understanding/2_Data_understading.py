
# coding: utf-8

# # 2. Data Understanding
# ## 2.1 Collect Initial Data
# ### Outputs:
# - Initial Data Collection Report
# 
# 

# ## 2.2 Describe Data
# ### Outputs:
# - Data Description Report 
# 
# [Data description.txt](./data_description.txt)
# 
# 

# ## 2.3 Explore Data
# ### Outputs:
# - Data Exploration Report

# In[2]:

import nltk
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np


# In[16]:

train= pd.read_csv("../data/train.csv")
train.set_index('Id')


# In[20]:

train['MSSubClass'].hist()


# In[28]:

train['MSZoning'].value_counts().plot(kind='bar')


# In[29]:

train['LotFrontage'].hist()


# ## 2.4 Verify Data Quality
# ### Outputs:
# 
# - Data Quality Report
