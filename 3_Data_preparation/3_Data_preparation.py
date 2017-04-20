
# coding: utf-8

# # 3. Data preparation
# 

# In[172]:

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


# In[173]:

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


# In[174]:

train= pd.read_csv("../data/train.csv")
test=pd.read_csv('../data/test.csv')


# ## 3.1 Select Data
# ### Outputs:
# - Rationale for Inclusion/Exclusion
# 

# In[ ]:




# 
# ## 3.2 Clean Data
# ### Outputs:
# - Data Cleaning Report
# 

# In[175]:

train=train[train['LotArea']<55000]
train=train[train['LotFrontage']<300]
train=train[train['MasVnrArea']<1200]
train=train[train['MasVnrArea']<1200]
train=train[train['BsmtFinSF1']<5000]
train=train[train['BsmtFinSF2']<1400]
train=train[train['TotalBsmtSF']<3500]
train['Electrical'].dropna(inplace=True)
train=train[train['1stFlrSF']!=0]
train=train[train['1stFlrSF']<4000]
train=train[train['GrLivArea']!=0]
train=train[train['WoodDeckSF']<750]
train=train[train['OpenPorchSF']<400]
train=train[train['EnclosedPorch']<400]


# In[176]:

dataset=pd.concat([train,test],keys=['train','test'])


# In[177]:

del dataset['Utilities']
del dataset['TotalBsmtSF']
del dataset['TotRmsAbvGrd']
del dataset['GarageYrBlt']
del dataset['GarageCars']


dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean(),inplace=True)
dataset['Alley'].fillna('XX',inplace=True)
dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean(),inplace=True)
dataset['BsmtQual'].fillna('XX',inplace=True)
dataset['BsmtCond'].fillna('XX',inplace=True)
dataset['BsmtExposure'].fillna('XX',inplace=True)
dataset['BsmtFinSF1'].fillna(0,inplace=True)
dataset['BsmtFinSF2'].fillna(0,inplace=True)
dataset['BsmtUnfSF'].fillna(0,inplace=True)
dataset['BsmtFinType1'].fillna('XX',inplace=True)
dataset['BsmtFinType2'].fillna('XX',inplace=True)

dataset['BsmtFullBath'].fillna(dataset['BsmtFullBath'].mean(),inplace=True)
dataset['BsmtHalfBath'].fillna(dataset['BsmtHalfBath'].mean(),inplace=True)
dataset['MiscFeature'].fillna('XX',inplace=True)
dataset['MiscVal'].fillna(0,inplace=True)
dataset['FireplaceQu'].fillna('XX',inplace=True)
dataset['GarageType'].fillna('XX',inplace=True)
dataset['GarageFinish'].fillna('XX',inplace=True)
dataset['GarageQual'].fillna('XX',inplace=True)
dataset['GarageCond'].fillna('XX',inplace=True)
dataset['GarageArea'].fillna(0,inplace=True)
dataset['PoolQC'].fillna('XX',inplace=True)
dataset['Fence'].fillna('XX',inplace=True)


# In[178]:

dataset.loc['train'].to_csv('../data/train_cleaned.csv')
dataset.loc['test'].to_csv('../data/test_cleaned.csv')


# In[187]:




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

# In[186]:

dataset['MSSubClass'] = dataset['MSSubClass'].apply(str)

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
                'Fireplaces',
                'GarageArea',
                'WoodDeckSF',
                'OpenPorchSF',
                'EnclosedPorch',
                '3SsnPorch',
                'ScreenPorch',
                'PoolArea',
                'MiscVal',
                'MoSold',
                'YrSold']
for i in Numeric_columns:
    dataset[i]=preprocessing.scale(dataset[i])
    
dataset=pd.get_dummies(dataset)
    
train_dummied=dataset.loc['train']
test_dummied=dataset.loc['test']

train_dummied=train_dummied.set_index('Id')
test_dummied=test_dummied.set_index('Id')

train_dummied.to_csv('../data/train_dummied.csv')
test_dummied.to_csv('../data/test_dummied.csv')
#correlation_matrix(dataset_dummied)


# In[ ]:




# In[ ]:



