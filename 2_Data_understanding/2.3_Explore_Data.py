
# coding: utf-8

# ## 2.3 Explore Data
# ### Outputs:
# - Data Exploration Report

# In[2]:

import nltk
import pandas as pd
import math
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn import datasets, linear_model
import numpy as np
from numbers import Number


# In[3]:

train= pd.read_csv("../data/train.csv")
train.set_index('Id')


# In[4]:

def hist_boxplot(column,figsize=(13,8)):
    fig = plt.figure(figsize=figsize) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,4])
    ax0 = plt.subplot(gs[0])
    ax0.grid(True)
    ax0.boxplot(column.dropna(),vert=False)
    ax1 = plt.subplot(gs[1])
    ax1.grid(True)
    ax1.hist(column.dropna())
    print (column.describe())
    print ('Null Values:',column.isnull().sum())
    
def hist_and_info(column,figsize=(13,4)):
    column.hist(figsize=figsize)
    print (column.describe())
    print ('Null Values:',column.isnull().sum())
    
def value_counts_and_info(column,figsize=(13,4)):
    column.value_counts().plot(kind='bar',figsize=figsize)
    print (column.value_counts())
    print ('Null Values:',column.isnull().sum())


# In[5]:

value_counts_and_info(train['MSSubClass'])


# In[6]:

value_counts_and_info(train['MSZoning'])


# In[7]:

#train['LotFrontage'].hist()
hist_boxplot(train['LotFrontage'])


# In[8]:

hist_boxplot(train['LotArea'])


# In[9]:

value_counts_and_info(train['Street'])


# In[10]:

## Se identificaron los valores NaN indica que no hay camino de entrada.
value_counts_and_info(train['Alley'])


# In[11]:

value_counts_and_info(train['LotShape'])


# In[12]:

value_counts_and_info(train['LandContour'])


# In[13]:

#value_counts_and_info(train['Utilities'])


# In[14]:

value_counts_and_info(train['LotConfig'])


# In[15]:

value_counts_and_info(train['LandSlope'])


# In[16]:

value_counts_and_info(train['Neighborhood'])


# In[17]:

value_counts_and_info(train['Condition1'])


# In[18]:

value_counts_and_info(train['Condition2'])


# In[19]:

value_counts_and_info(train['BldgType'])


# In[20]:

value_counts_and_info(train['HouseStyle'])


# In[21]:

value_counts_and_info(train['OverallQual'])


# In[22]:

value_counts_and_info(train['OverallCond'])


# In[23]:

hist_and_info(train['YearBuilt'])


# In[24]:

hist_and_info(train['YearRemodAdd'])


# In[25]:

value_counts_and_info(train['RoofStyle'])


# In[26]:

value_counts_and_info(train['RoofMatl'])


# In[27]:

value_counts_and_info(train['Exterior1st'])


# In[28]:

value_counts_and_info(train['Exterior2nd'])


# In[29]:

value_counts_and_info(train['MasVnrType'])


# In[30]:

hist_boxplot(train['MasVnrArea'])


# In[31]:

value_counts_and_info(train['ExterQual'])


# In[32]:

value_counts_and_info(train['ExterCond'])


# In[33]:

value_counts_and_info(train['Foundation'])


# In[34]:

value_counts_and_info(train['BsmtQual'])


# In[35]:

value_counts_and_info(train['BsmtCond'])


# In[36]:

value_counts_and_info(train['BsmtExposure'])


# In[37]:

value_counts_and_info(train['BsmtFinType1'])


# In[38]:

hist_boxplot(train['BsmtFinSF1'])


# In[39]:

value_counts_and_info(train['BsmtFinType2'])


# In[40]:

hist_boxplot(train['BsmtFinSF2'])


# In[41]:

hist_and_info(train['BsmtUnfSF'])


# In[42]:

hist_boxplot(train['TotalBsmtSF'])


# In[43]:

value_counts_and_info(train['Heating'])


# In[44]:

value_counts_and_info(train['HeatingQC'])


# In[45]:

value_counts_and_info(train['CentralAir'])


# In[46]:

value_counts_and_info(train['Electrical'])


# In[47]:

hist_boxplot(train['1stFlrSF'])


# In[48]:

hist_boxplot(train['2ndFlrSF'])


# In[49]:

hist_boxplot(train['LowQualFinSF'])


# In[50]:

hist_boxplot(train['GrLivArea'])


# In[51]:

value_counts_and_info(train['BsmtFullBath'])


# In[52]:

value_counts_and_info(train['BsmtHalfBath'])


# In[53]:

value_counts_and_info(train['FullBath'])


# In[54]:

value_counts_and_info(train['HalfBath'])


# In[55]:

value_counts_and_info(train['BedroomAbvGr'])


# In[56]:

value_counts_and_info(train['KitchenAbvGr'])


# In[57]:

value_counts_and_info(train['KitchenQual'])


# In[58]:

value_counts_and_info(train['TotRmsAbvGrd'])


# In[59]:

value_counts_and_info(train['Functional'])


# In[60]:

value_counts_and_info(train['Fireplaces'])


# In[61]:

value_counts_and_info(train['FireplaceQu'])


# In[62]:

value_counts_and_info(train['GarageType'])


# In[63]:

hist_and_info(train['GarageYrBlt'])


# In[64]:

value_counts_and_info(train['GarageFinish'])


# In[65]:

value_counts_and_info(train['GarageCars'])


# In[66]:

hist_boxplot(train['GarageArea'])


# In[67]:

value_counts_and_info(train['GarageQual'])


# In[68]:

value_counts_and_info(train['GarageCond'])


# In[69]:

value_counts_and_info(train['PavedDrive'])


# In[70]:

hist_boxplot(train['WoodDeckSF'])


# In[71]:

hist_boxplot(train['OpenPorchSF'])


# In[72]:

hist_boxplot(train['EnclosedPorch'])


# In[73]:

hist_boxplot(train['3SsnPorch'])


# In[74]:

hist_and_info(train['ScreenPorch'])


# In[75]:

hist_and_info(train['PoolArea'])


# In[76]:

value_counts_and_info(train['PoolQC'])


# In[77]:

value_counts_and_info(train['Fence'])


# In[78]:

value_counts_and_info(train['MiscFeature'])


# In[79]:

value_counts_and_info(train['MiscVal'])


# In[80]:

value_counts_and_info(train['MoSold'])


# In[81]:

value_counts_and_info(train['YrSold'])


# In[82]:

value_counts_and_info(train['SaleType'])


# In[83]:

value_counts_and_info(train['SaleCondition'])


# 
# 

# ## 2.4 Verify Data Quality
# ### Outputs:
# 
# - Data Quality Report
