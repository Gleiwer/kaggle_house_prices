
# coding: utf-8

# ## 2.3 Explore Data
# ### Outputs:
# - Data Exploration Report

# In[954]:

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


# In[1036]:

train= pd.read_csv("../data/train.csv")
train.set_index('Id')


# In[956]:

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


# In[957]:

hist_and_info(train['MSSubClass'])


# In[958]:

value_counts_and_info(train['MSZoning'])


# In[959]:

#train['LotFrontage'].hist()
hist_boxplot(train['LotFrontage'])


# In[960]:

hist_boxplot(train['LotArea'])


# In[961]:

value_counts_and_info(train['Street'])


# In[962]:

## Se identificaron los valores NaN indica que no hay camino de entrada.
value_counts_and_info(train['Alley'])


# In[963]:

value_counts_and_info(train['LotShape'])


# In[964]:

value_counts_and_info(train['LandContour'])


# In[965]:

#value_counts_and_info(train['Utilities'])


# In[966]:

value_counts_and_info(train['LotConfig'])


# In[967]:

value_counts_and_info(train['LandSlope'])


# In[968]:

value_counts_and_info(train['Neighborhood'])


# In[969]:

value_counts_and_info(train['Condition1'])


# In[970]:

value_counts_and_info(train['Condition2'])


# In[971]:

value_counts_and_info(train['BldgType'])


# In[972]:

value_counts_and_info(train['HouseStyle'])


# In[973]:

value_counts_and_info(train['OverallQual'])


# In[974]:

value_counts_and_info(train['OverallCond'])


# In[975]:

hist_and_info(train['YearBuilt'])


# In[976]:

hist_and_info(train['YearRemodAdd'])


# In[977]:

value_counts_and_info(train['RoofStyle'])


# In[978]:

value_counts_and_info(train['RoofMatl'])


# In[979]:

value_counts_and_info(train['Exterior1st'])


# In[980]:

value_counts_and_info(train['Exterior2nd'])


# In[981]:

value_counts_and_info(train['MasVnrType'])


# In[982]:

hist_boxplot(train['MasVnrArea'])


# In[983]:

value_counts_and_info(train['ExterQual'])


# In[984]:

value_counts_and_info(train['ExterCond'])


# In[985]:

value_counts_and_info(train['Foundation'])


# In[986]:

value_counts_and_info(train['BsmtQual'])


# In[987]:

value_counts_and_info(train['BsmtCond'])


# In[988]:

value_counts_and_info(train['BsmtExposure'])


# In[989]:

value_counts_and_info(train['BsmtFinType1'])


# In[990]:

hist_boxplot(train['BsmtFinSF1'])


# In[991]:

value_counts_and_info(train['BsmtFinType2'])


# In[992]:

hist_boxplot(train['BsmtFinSF2'])


# In[993]:

hist_and_info(train['BsmtUnfSF'])


# In[994]:

hist_boxplot(train['TotalBsmtSF'])


# In[995]:

value_counts_and_info(train['Heating'])


# In[996]:

value_counts_and_info(train['HeatingQC'])


# In[997]:

value_counts_and_info(train['CentralAir'])


# In[998]:

value_counts_and_info(train['Electrical'])


# In[999]:

hist_boxplot(train['1stFlrSF'])


# In[1000]:

hist_boxplot(train['2ndFlrSF'])


# In[1001]:

hist_boxplot(train['LowQualFinSF'])


# In[1002]:

hist_boxplot(train['GrLivArea'])


# In[1003]:

value_counts_and_info(train['BsmtFullBath'])


# In[1004]:

value_counts_and_info(train['BsmtHalfBath'])


# In[1005]:

value_counts_and_info(train['FullBath'])


# In[1006]:

value_counts_and_info(train['HalfBath'])


# In[1007]:

value_counts_and_info(train['BedroomAbvGr'])


# In[1008]:

value_counts_and_info(train['KitchenAbvGr'])


# In[1009]:

value_counts_and_info(train['KitchenQual'])


# In[1010]:

value_counts_and_info(train['TotRmsAbvGrd'])


# In[1011]:

value_counts_and_info(train['Functional'])


# In[1012]:

value_counts_and_info(train['Fireplaces'])


# In[1013]:

value_counts_and_info(train['FireplaceQu'])


# In[1014]:

value_counts_and_info(train['GarageType'])


# In[1015]:

hist_and_info(train['GarageYrBlt'])


# In[1016]:

value_counts_and_info(train['GarageFinish'])


# In[1017]:

value_counts_and_info(train['GarageCars'])


# In[1018]:

hist_boxplot(train['GarageArea'])


# In[1019]:

value_counts_and_info(train['GarageQual'])


# In[1020]:

value_counts_and_info(train['GarageCond'])


# In[1021]:

value_counts_and_info(train['PavedDrive'])


# In[1022]:

hist_boxplot(train['WoodDeckSF'])


# In[1023]:

hist_boxplot(train['OpenPorchSF'])


# In[1024]:

hist_boxplot(train['EnclosedPorch'])


# In[1025]:

hist_boxplot(train['3SsnPorch'])


# In[1026]:

hist_and_info(train['ScreenPorch'])


# In[1027]:

hist_and_info(train['PoolArea'])


# In[1028]:

value_counts_and_info(train['PoolQC'])


# In[1029]:

value_counts_and_info(train['Fence'])


# In[1030]:

value_counts_and_info(train['MiscFeature'])


# In[1031]:

value_counts_and_info(train['MiscVal'])


# In[1032]:

value_counts_and_info(train['MoSold'])


# In[1033]:

value_counts_and_info(train['YrSold'])


# In[1034]:

value_counts_and_info(train['SaleType'])


# In[1035]:

value_counts_and_info(train['SaleCondition'])


# 
# 

# ## 2.4 Verify Data Quality
# ### Outputs:
# 
# - Data Quality Report
