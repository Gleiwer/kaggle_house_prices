
# coding: utf-8

# # 3. Data preparation
# 

# In[92]:

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


# In[93]:

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


# In[94]:

train= pd.read_csv("../data/train.csv")
test=pd.read_csv('../data/test.csv')
dataset=pd.concat([train,test],keys=['train','test'])



# ## 3.1 Select Data
# ### Outputs:
# - Rationale for Inclusion/Exclusion
# 

# In[95]:

del dataset['Utilities']


# 
# ## 3.2 Clean Data
# ### Outputs:
# - Data Cleaning Report
# 

# In[ ]:




# In[96]:

dataset=dataset[dataset['LotArea']<55000]
dataset=dataset[dataset['LotFrontage']<300]
dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean(),inplace=True)
dataset=dataset[dataset['MasVnrArea']<1200]
dataset['Alley'].fillna('XX',inplace=True)
dataset=dataset[dataset['MasVnrArea']<1200]
dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean(),inplace=True)
dataset['BsmtQual'].fillna('XX',inplace=True)
dataset['BsmtCond'].fillna('XX',inplace=True)
dataset['BsmtExposure'].fillna('XX',inplace=True)
dataset['BsmtFinType1'].fillna('XX',inplace=True)
dataset['BsmtFinType2'].fillna('XX',inplace=True)
dataset=dataset[dataset['BsmtFinSF1']<5000]
dataset=dataset[dataset['BsmtFinSF2']<1400]
dataset=dataset[dataset['TotalBsmtSF']<3500]
dataset['Electrical'].dropna(inplace=True)
dataset=dataset[dataset['1stFlrSF']!=0]
dataset=dataset[dataset['1stFlrSF']<4000]
dataset=dataset[dataset['GrLivArea']!=0]
dataset['MiscFeature'].fillna('XX',inplace=True)
dataset['MiscVal'].fillna(0,inplace=True)
dataset['FireplaceQu'].fillna('XX',inplace=True)
dataset['GarageType'].fillna('XX',inplace=True)
dataset['GarageYrBlt'].fillna(0,inplace=True)
dataset['GarageFinish'].fillna('XX',inplace=True)
dataset['GarageQual'].fillna('XX',inplace=True)
dataset['GarageCond'].fillna('XX',inplace=True)
dataset=dataset[dataset['WoodDeckSF']<750]
dataset=dataset[dataset['OpenPorchSF']<400]
dataset=dataset[dataset['EnclosedPorch']<400]
dataset['PoolQC'].fillna('XX',inplace=True)
dataset['Fence'].fillna('XX',inplace=True)


# In[97]:

dataset.loc['train'].to_csv('../data/train_cleaned.csv')
dataset.loc['test'].to_csv('../data/test_cleaned.csv')


# In[99]:




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

# In[ ]:

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
    #pdb.set_trace()
    dataset[i]=preprocessing.scale(dataset[i])
    print (i,'ok')

dataset_dummied=pd.get_dummies(dataset)
dataset_dummied=dataset_dummied.set_index('Id')
dataset_dummied.loc['train'].to_csv('../data/train_dummied.csv')
dataset_dummied.loc['test'].to_csv('../data/test_dummied.csv')
correlation_matrix(dataset_dummied)


# In[ ]:

train_dummied

