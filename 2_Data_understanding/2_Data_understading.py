
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


# In[30]:

train['LotFrontage'].hist()


# In[117]:

f, axarr = plt.subplots(2, sharex=True,figsize=(13,8))
plt.setp(axarr, xticks=[x*20000 for x in range(13)])
axarr[1].grid(True)
axarr[0].grid(True)

axarr[1].hist(train['LotArea'])
axarr[0].boxplot(train['LotArea'],vert=False)


# In[118]:

train['Street'].value_counts().plot(kind='bar')
train['Street'].value_counts()


# In[120]:

train['Alley'].value_counts().plot(kind='bar')
train['Alley'].value_counts()


# In[122]:

train['LotShape'].value_counts().plot(kind='bar')
train['LotShape'].value_counts()


# In[124]:

train['LandContour'].value_counts().plot(kind='bar')
train['LandContour'].value_counts()


# In[126]:

train['Utilities'].value_counts().plot(kind='bar')
train['Utilities'].value_counts()


# In[127]:

train['LotConfig'].value_counts().plot(kind='bar')
train['LotConfig'].value_counts()


# In[128]:

train['LandSlope'].value_counts().plot(kind='bar')
train['LandSlope'].value_counts()


# In[130]:

train['Neighborhood'].value_counts().plot(kind='bar')
train['Neighborhood'].value_counts()


# In[132]:

train['Condition1'].value_counts().plot(kind='bar')
train['Condition1'].value_counts()


# In[133]:

train['Condition2'].value_counts().plot(kind='bar')
train['Condition2'].value_counts()


# In[134]:

train['BldgType'].value_counts().plot(kind='bar')
train['BldgType'].value_counts()


# In[135]:

train['HouseStyle'].value_counts().plot(kind='bar')
train['HouseStyle'].value_counts()


# In[137]:

train['OverallQual'].value_counts().plot(kind='bar')
train['OverallQual'].value_counts()


# In[139]:

train['OverallCond'].value_counts().plot(kind='bar')
train['OverallCond'].value_counts()


# In[141]:

train['YearBuilt'].hist()


# In[142]:

train['YearRemodAdd'].hist()


# ## 2.4 Verify Data Quality
# ### Outputs:
# 
# - Data Quality Report
