
# coding: utf-8

# # 3. Data preparation
# 

# In[62]:

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

from sklearn import preprocessing


# In[63]:

def correlation_matrix(df,figsize=(15,15)):
    from matplotlib import pyplot as plt
    from matplotlib import cm as cm

    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 30)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])
    plt.show()


# In[64]:

train= pd.read_csv("../data/train.csv")
train.set_index('Id')


# ## 3.1 Select Data
# ### Outputs:
# - Rationale for Inclusion/Exclusion
# 

# In[65]:

del train['Utilities']


# 
# ## 3.2 Clean Data
# ### Outputs:
# - Data Cleaning Report
# 

# In[ ]:




# In[66]:

train=train[train['LotArea']<55000]
train=train[train['LotFrontage']<300]
train['LotFrontage'].fillna(train['LotFrontage'].mean(),inplace=True)
train=train[train['MasVnrArea']<1200]
train['Alley'].fillna('XX',inplace=True)
train=train[train['MasVnrArea']<1200]
train['MasVnrArea'].fillna(train['MasVnrArea'].mean(),inplace=True)
train['BsmtQual'].fillna('XX',inplace=True)
train['BsmtCond'].fillna('XX',inplace=True)
train['BsmtExposure'].fillna('XX',inplace=True)
train['BsmtFinType1'].fillna('XX',inplace=True)
train['BsmtFinType2'].fillna('XX',inplace=True)
train=train[train['BsmtFinSF1']<5000]
train=train[train['BsmtFinSF2']<1400]
train=train[train['TotalBsmtSF']<3500]
train['Electrical'].dropna(inplace=True)
train=train[train['1stFlrSF']!=0]
train=train[train['1stFlrSF']<4000]
train=train[train['GrLivArea']!=0]
train['MiscFeature'].fillna('XX',inplace=True)
train['MiscVal'].fillna(0,inplace=True)
train['FireplaceQu'].fillna('XX',inplace=True)
train['GarageType'].fillna('XX',inplace=True)
train['GarageYrBlt'].fillna(0,inplace=True)
train['GarageFinish'].fillna('XX',inplace=True)
train['GarageQual'].fillna('XX',inplace=True)
train['GarageCond'].fillna('XX',inplace=True)
train=train[train['WoodDeckSF']<750]
train=train[train['OpenPorchSF']<400]
train=train[train['EnclosedPorch']<400]
train['PoolQC'].fillna('XX',inplace=True)
train['Fence'].fillna('XX',inplace=True)


# In[67]:

train.to_csv('../data/train_cleaned.csv')


# In[ ]:




# 
# ## 3.3 Construct Data
# ### Outputs:
# - Derived Attributes
# - Generated Records
# 
# ## 3.4 Integrate Data
# ### Outputs:
# 
# - Merged Data
# 
# 

# ## 3.5 Format Data
# ### Outputs:
# 
# - Reformatted Data
# - Dataset
# - Dataset Description

# In[68]:

train['MSSubClass'] = train['MSSubClass'].apply(str)

Numeric_columns=['LotFrontage',
                'LotArea',
                'OverallQual',
                'OverallCond',
                'YearBuilt',
                'YearRemodAdd',
                'MasVnrArea',
                'BsmtFinSF1',
                'BsmtFinSF2',
                'BsmtUnfSF',
                'TotalBsmtSF',
                '1stFlrSF',
                '2ndFlrSF',
                'LowQualFinSF',
                'GrLivArea',
                'BsmtFullBath',
                'BsmtHalfBath',
                'FullBath',
                'HalfBath',
                'BedroomAbvGr',
                'KitchenAbvGr',
                'TotRmsAbvGrd',
                'Fireplaces',
                'GarageYrBlt',
                'GarageCars',
                'GarageArea',
                'WoodDeckSF',
                'OpenPorchSF',
                'EnclosedPorch',
                '3SsnPorch',
                'ScreenPorch',
                'PoolArea',
                'MiscVal',
                'MoSold',
                'YrSold',
                'TotalBsmtSF']
for i in Numeric_columns:
    train[i]=preprocessing.scale(train[i])
    train[i]=preprocessing.

train_dummied=pd.get_dummies(train)
train_dummied=train_dummied.set_index('Id')
train_dummied.to_csv('../data/train_dummied.csv')
correlation_matrix(train_dummied)


# In[70]:

train_dummied


# In[69]:

test_dummied=pd.read_csv('../data/test.csv')
test_dummied

