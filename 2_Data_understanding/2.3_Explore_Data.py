
# coding: utf-8

# ## 2.3 Explore Data
# ### Outputs:
# - Data Exploration Report

# In[127]:

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

import seaborn as sns


# In[133]:

train= pd.read_csv("../data/train.csv")
train=train.set_index('Id')


# In[134]:

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


# In[135]:

value_counts_and_info(train['MSSubClass'])


# In[131]:

value_counts_and_info(train['MSZoning'])


# In[136]:

#train['LotFrontage'].hist()
hist_boxplot(train['LotFrontage'])


# In[137]:

hist_boxplot(train['LotArea'])


# In[138]:

value_counts_and_info(train['Street'])


# In[ ]:

## Se identificaron los valores NaN indica que no hay camino de entrada.
value_counts_and_info(train['Alley'])


# In[ ]:

value_counts_and_info(train['LotShape'])


# In[ ]:

value_counts_and_info(train['LandContour'])


# In[ ]:

#value_counts_and_info(train['Utilities'])


# In[ ]:

value_counts_and_info(train['LotConfig'])


# In[ ]:

value_counts_and_info(train['LandSlope'])


# In[ ]:

value_counts_and_info(train['Neighborhood'])


# In[ ]:

value_counts_and_info(train['Condition1'])


# In[ ]:

value_counts_and_info(train['Condition2'])


# In[ ]:

value_counts_and_info(train['BldgType'])


# In[ ]:

value_counts_and_info(train['HouseStyle'])


# In[ ]:

value_counts_and_info(train['OverallQual'])


# In[ ]:

value_counts_and_info(train['OverallCond'])


# In[ ]:

hist_and_info(train['YearBuilt'])


# In[ ]:

hist_and_info(train['YearRemodAdd'])


# In[ ]:

value_counts_and_info(train['RoofStyle'])


# In[ ]:

value_counts_and_info(train['RoofMatl'])


# In[ ]:

value_counts_and_info(train['Exterior1st'])


# In[ ]:

value_counts_and_info(train['Exterior2nd'])


# In[ ]:

value_counts_and_info(train['MasVnrType'])


# In[ ]:

hist_boxplot(train['MasVnrArea'])


# In[ ]:

value_counts_and_info(train['ExterQual'])


# In[ ]:

value_counts_and_info(train['ExterCond'])


# In[ ]:

value_counts_and_info(train['Foundation'])


# In[ ]:

value_counts_and_info(train['BsmtQual'])


# In[ ]:

value_counts_and_info(train['BsmtCond'])


# In[ ]:

value_counts_and_info(train['BsmtExposure'])


# In[ ]:

value_counts_and_info(train['BsmtFinType1'])


# In[ ]:

hist_boxplot(train['BsmtFinSF1'])


# In[ ]:

value_counts_and_info(train['BsmtFinType2'])


# In[ ]:

hist_boxplot(train['BsmtFinSF2'])


# In[ ]:

hist_and_info(train['BsmtUnfSF'])


# In[ ]:

hist_boxplot(train['TotalBsmtSF'])


# In[ ]:

value_counts_and_info(train['Heating'])


# In[ ]:

value_counts_and_info(train['HeatingQC'])


# In[ ]:

value_counts_and_info(train['CentralAir'])


# In[ ]:

value_counts_and_info(train['Electrical'])


# In[ ]:

hist_boxplot(train['1stFlrSF'])


# In[ ]:

hist_boxplot(train['2ndFlrSF'])


# In[ ]:

hist_boxplot(train['LowQualFinSF'])


# In[ ]:

hist_boxplot(train['GrLivArea'])


# In[ ]:

value_counts_and_info(train['BsmtFullBath'])


# In[ ]:

value_counts_and_info(train['BsmtHalfBath'])


# In[ ]:

value_counts_and_info(train['FullBath'])


# In[ ]:

value_counts_and_info(train['HalfBath'])


# In[ ]:

value_counts_and_info(train['BedroomAbvGr'])


# In[ ]:

value_counts_and_info(train['KitchenAbvGr'])


# In[ ]:

value_counts_and_info(train['KitchenQual'])


# In[ ]:

value_counts_and_info(train['TotRmsAbvGrd'])


# In[ ]:

value_counts_and_info(train['Functional'])


# In[ ]:

value_counts_and_info(train['Fireplaces'])


# In[ ]:

value_counts_and_info(train['FireplaceQu'])


# In[ ]:

value_counts_and_info(train['GarageType'])


# In[ ]:

hist_and_info(train['GarageYrBlt'])


# In[ ]:

value_counts_and_info(train['GarageFinish'])


# In[ ]:

value_counts_and_info(train['GarageCars'])


# In[ ]:

hist_boxplot(train['GarageArea'])


# In[ ]:

value_counts_and_info(train['GarageQual'])


# In[ ]:

value_counts_and_info(train['GarageCond'])


# In[ ]:

value_counts_and_info(train['PavedDrive'])


# In[ ]:

hist_boxplot(train['WoodDeckSF'])


# In[ ]:

hist_boxplot(train['OpenPorchSF'])


# In[ ]:

hist_boxplot(train['EnclosedPorch'])


# In[ ]:

hist_boxplot(train['3SsnPorch'])


# In[ ]:

hist_and_info(train['ScreenPorch'])


# In[ ]:

hist_and_info(train['PoolArea'])


# In[ ]:

value_counts_and_info(train['PoolQC'])


# In[ ]:

value_counts_and_info(train['Fence'])


# In[ ]:

value_counts_and_info(train['MiscFeature'])


# In[ ]:

value_counts_and_info(train['MiscVal'])


# In[ ]:

value_counts_and_info(train['MoSold'])


# In[ ]:

value_counts_and_info(train['YrSold'])


# In[ ]:

value_counts_and_info(train['SaleType'])


# In[ ]:

value_counts_and_info(train['SaleCondition'])


# In[139]:

corrmat = train.iloc[:,:-1].corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);


# In[ ]:




# ## 2.4 Verify Data Quality
# ### Outputs:
# 
# - Data Quality Report
