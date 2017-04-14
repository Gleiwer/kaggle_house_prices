
# coding: utf-8

# ## 2.3 Explore Data
# ### Outputs:
# - Data Exploration Report

# In[632]:

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


# In[633]:

train= pd.read_csv("../data/train.csv")
train.set_index('Id')


# In[634]:

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


# In[635]:

hist_and_info(train['MSSubClass'])


# In[636]:

value_counts_and_info(train['MSZoning'])


# In[637]:

#train['LotFrontage'].hist()
hist_boxplot(train['LotFrontage'])


# In[638]:

hist_boxplot(train['LotArea'])


# In[639]:

value_counts_and_info(train['Street'])


# In[640]:

## Se identificaron los valores NaN indica que no hay camino de entrada.
value_counts_and_info(train['Alley'])


# In[641]:

value_counts_and_info(train['LotShape'])


# In[642]:

value_counts_and_info(train['LandContour'])


# In[643]:

value_counts_and_info(train['Utilities'])


# In[644]:

value_counts_and_info(train['LotConfig'])


# In[645]:

value_counts_and_info(train['LandSlope'])


# In[646]:

value_counts_and_info(train['Neighborhood'])


# In[647]:

value_counts_and_info(train['Condition1'])


# In[648]:

value_counts_and_info(train['Condition2'])


# In[649]:

value_counts_and_info(train['BldgType'])


# In[650]:

value_counts_and_info(train['HouseStyle'])


# In[651]:

value_counts_and_info(train['OverallQual'])


# In[652]:

value_counts_and_info(train['OverallCond'])


# In[653]:

hist_and_info(train['YearBuilt'])


# In[654]:

hist_and_info(train['YearRemodAdd'])


# In[655]:

value_counts_and_info(train['RoofStyle'])


# In[656]:

value_counts_and_info(train['RoofMatl'])


# In[657]:

value_counts_and_info(train['Exterior1st'])


# In[658]:

value_counts_and_info(train['Exterior2nd'])


# In[659]:

value_counts_and_info(train['MasVnrType'])


# In[660]:

hist_boxplot(train['MasVnrArea'])


# In[661]:

value_counts_and_info(train['ExterQual'])


# In[662]:

value_counts_and_info(train['ExterCond'])


# In[663]:

value_counts_and_info(train['Foundation'])


# In[664]:

value_counts_and_info(train['BsmtQual'])


# In[665]:

value_counts_and_info(train['BsmtCond'])


# In[666]:

value_counts_and_info(train['BsmtExposure'])


# In[667]:

value_counts_and_info(train['BsmtFinType1'])


# In[668]:

hist_boxplot(train['BsmtFinSF1'])


# In[669]:

value_counts_and_info(train['BsmtFinType2'])


# In[670]:

hist_boxplot(train['BsmtFinSF2'])


# In[671]:

hist_and_info(train['BsmtUnfSF'])


# In[672]:

hist_boxplot(train['TotalBsmtSF'])


# In[673]:

value_counts_and_info(train['Heating'])


# In[674]:

value_counts_and_info(train['HeatingQC'])


# In[675]:

value_counts_and_info(train['CentralAir'])


# In[676]:

value_counts_and_info(train['Electrical'])


# In[677]:

hist_boxplot(train['1stFlrSF'])


# In[678]:

hist_boxplot(train['2ndFlrSF'])


# In[679]:

hist_boxplot(train['LowQualFinSF'])


# In[680]:

hist_boxplot(train['GrLivArea'])


# In[714]:

value_counts_and_info(train['BsmtFullBath'])


# In[682]:

value_counts_and_info(train['BsmtHalfBath'])


# In[683]:

value_counts_and_info(train['FullBath'])


# In[684]:

value_counts_and_info(train['HalfBath'])


# In[685]:

value_counts_and_info(train['BedroomAbvGr'])


# In[686]:

value_counts_and_info(train['KitchenAbvGr'])


# In[687]:

value_counts_and_info(train['KitchenQual'])


# In[688]:

value_counts_and_info(train['TotRmsAbvGrd'])


# In[689]:

value_counts_and_info(train['Functional'])


# In[690]:

value_counts_and_info(train['Fireplaces'])


# In[691]:

value_counts_and_info(train['FireplaceQu'])


# In[692]:

value_counts_and_info(train['GarageType'])


# In[693]:

hist_and_info(train['GarageYrBlt'])


# In[694]:

value_counts_and_info(train['GarageFinish'])


# In[695]:

value_counts_and_info(train['GarageCars'])


# In[715]:

hist_boxplot(train['GarageArea'])


# In[697]:

value_counts_and_info(train['GarageQual'])


# In[698]:

value_counts_and_info(train['GarageCond'])


# In[699]:

value_counts_and_info(train['PavedDrive'])


# In[700]:

hist_boxplot(train['WoodDeckSF'])


# In[701]:

hist_boxplot(train['OpenPorchSF'])


# In[702]:

hist_boxplot(train['EnclosedPorch'])


# In[703]:

hist_boxplot(train['3SsnPorch'])


# In[704]:

hist_and_info(train['ScreenPorch'])


# In[705]:

hist_and_info(train['PoolArea'])


# In[706]:

value_counts_and_info(train['PoolQC'])


# In[707]:

value_counts_and_info(train['Fence'])


# In[708]:

value_counts_and_info(train['MiscFeature'])


# In[709]:

value_counts_and_info(train['MiscVal'])


# In[710]:

value_counts_and_info(train['MoSold'])


# In[711]:

value_counts_and_info(train['YrSold'])


# In[712]:

value_counts_and_info(train['SaleType'])


# In[713]:

value_counts_and_info(train['SaleCondition'])


# 
# 

# ## 2.4 Verify Data Quality
# ### Outputs:
# 
# - Data Quality Report
