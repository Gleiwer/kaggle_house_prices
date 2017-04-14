
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

# In[222]:

import nltk
import pandas as pd
import math
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np

from numbers import Number


# In[223]:

train= pd.read_csv("../data/train.csv")
train.set_index('Id')


# In[224]:

## Generalizar la visualización dependiendo del tipo de variable.
f, axarr = plt.subplots(int(round(math.sqrt(len(train.columns)))),
                        int(round(math.sqrt(len(train.columns)))),
                        figsize=(16,13))
for i,j in enumerate(train.columns):
    if train[j].dtype in (np.int64,np.float64):
        axarr[i].hist(train[j])
        


# In[225]:

train['MSSubClass'].hist()


# In[226]:

train['MSZoning'].value_counts().plot(kind='bar')


# In[227]:

train['LotFrontage'].hist()


# In[228]:

f, axarr = plt.subplots(2, sharex=True,figsize=(13,8))
plt.setp(axarr, xticks=[x*20000 for x in range(13)])
axarr[1].grid(True)
axarr[0].grid(True)

axarr[1].hist(train['LotArea'])
axarr[0].boxplot(train['LotArea'],vert=False)


# In[229]:

train['Street'].value_counts().plot(kind='bar')
train['Street'].value_counts()


# In[230]:

train['Alley'].value_counts().plot(kind='bar')
train['Alley'].value_counts()


# In[231]:

train['LotShape'].value_counts().plot(kind='bar')
train['LotShape'].value_counts()


# In[232]:

train['LandContour'].value_counts().plot(kind='bar')
train['LandContour'].value_counts()


# In[233]:

train['Utilities'].value_counts().plot(kind='bar')
train['Utilities'].value_counts()


# In[234]:

train['LotConfig'].value_counts().plot(kind='bar')
train['LotConfig'].value_counts()


# In[235]:

train['LandSlope'].value_counts().plot(kind='bar')
train['LandSlope'].value_counts()


# In[236]:

train['Neighborhood'].value_counts().plot(kind='bar')
train['Neighborhood'].value_counts()


# In[237]:

train['Condition1'].value_counts().plot(kind='bar')
train['Condition1'].value_counts()


# In[238]:

train['Condition2'].value_counts().plot(kind='bar')
train['Condition2'].value_counts()


# In[239]:

train['BldgType'].value_counts().plot(kind='bar')
train['BldgType'].value_counts()


# In[240]:

train['HouseStyle'].value_counts().plot(kind='bar')
train['HouseStyle'].value_counts()


# In[241]:

train['OverallQual'].value_counts().plot(kind='bar')
train['OverallQual'].value_counts()


# In[242]:

train['OverallCond'].value_counts().plot(kind='bar')
train['OverallCond'].value_counts()


# In[243]:

train['YearBuilt'].hist()


# In[244]:

train['YearRemodAdd'].hist()


# In[245]:

train['RoofStyle'].value_counts().plot(kind='bar')
train['RoofStyle'].value_counts()


# In[246]:

train['RoofMatl'].value_counts().plot(kind='bar')
train['RoofMatl'].value_counts()


# In[247]:

train['Exterior1st'].value_counts().plot(kind='bar')
train['Exterior1st'].value_counts()


# In[248]:

train['Exterior2nd'].value_counts().plot(kind='bar')
train['Exterior2nd'].value_counts()


# In[249]:

train['MasVnrType'].value_counts().plot(kind='bar')
train['MasVnrType'].value_counts()


# In[250]:

train['MasVnrArea'].hist()


# In[251]:

train['ExterQual'].value_counts().plot(kind='bar')
train['ExterQual'].value_counts()


# In[252]:

train['ExterCond'].value_counts().plot(kind='bar')
train['ExterCond'].value_counts()


# In[253]:

train['Foundation'].value_counts().plot(kind='bar')
train['Foundation'].value_counts()


# In[254]:

train['BsmtQual'].value_counts().plot(kind='bar')
train['BsmtQual'].value_counts()


# In[255]:

train['BsmtCond'].value_counts().plot(kind='bar')
train['BsmtCond'].value_counts()


# In[256]:

train['BsmtExposure'].value_counts().plot(kind='bar')
train['BsmtExposure'].value_counts()


# In[257]:

train['BsmtFinType1'].value_counts().plot(kind='bar')
train['BsmtFinType1'].value_counts()


# In[258]:

f, axarr = plt.subplots(2, sharex=True,figsize=(13,8))
axarr[1].grid(True)
axarr[0].grid(True)
axarr[1].hist(train['BsmtFinSF1'])
axarr[0].boxplot(train['BsmtFinSF1'],vert=False)


# In[259]:

train['BsmtFinType2'].value_counts().plot(kind='bar')
train['BsmtFinType2'].value_counts()


# In[260]:

train['BsmtFinSF2'].hist()


# In[261]:

train['BsmtUnfSF'].hist()


# In[262]:

train['TotalBsmtSF'].hist()


# In[263]:

train['Heating'].value_counts().plot(kind='bar')
train['Heating'].value_counts()


# In[264]:

train['HeatingQC'].value_counts().plot(kind='bar')
train['HeatingQC'].value_counts()


# In[265]:

train['CentralAir'].value_counts().plot(kind='bar')
train['CentralAir'].value_counts()


# In[266]:

train['Electrical'].value_counts().plot(kind='bar')
train['Electrical'].value_counts()


# In[267]:

f, axarr = plt.subplots(2, sharex=True,figsize=(13,8))
axarr[1].grid(True)
axarr[0].grid(True)
axarr[1].hist(train['1stFlrSF'])
axarr[0].boxplot(train['1stFlrSF'],vert=False)


# In[268]:

train['2ndFlrSF'].hist()


# In[269]:

f, axarr = plt.subplots(2, sharex=True,figsize=(13,8))
axarr[1].grid(True)
axarr[0].grid(True)
axarr[1].hist(train['LowQualFinSF'])
axarr[0].boxplot(train['LowQualFinSF'],vert=False)


# In[270]:

f, axarr = plt.subplots(2, sharex=True,figsize=(13,8))
axarr[1].grid(True)
axarr[0].grid(True)
axarr[1].hist(train['GrLivArea'])
axarr[0].boxplot(train['GrLivArea'],vert=False)


# In[271]:

train['BsmtFullBath'].value_counts().plot(kind='bar')
train['BsmtFullBath'].value_counts()


# In[272]:

train['BsmtHalfBath'].value_counts().plot(kind='bar')
train['BsmtHalfBath'].value_counts()


# In[273]:

train['FullBath'].value_counts().plot(kind='bar')
train['FullBath'].value_counts()


# In[274]:

train['HalfBath'].value_counts().plot(kind='bar')
train['HalfBath'].value_counts()


# In[277]:

train['BedroomAbvGr'].value_counts().plot(kind='bar')
train['BedroomAbvGr'].value_counts()


# In[279]:

train['KitchenAbvGr'].value_counts().plot(kind='bar')
train['KitchenAbvGr'].value_counts()


# In[281]:

train['KitchenQual'].value_counts().plot(kind='bar')
train['KitchenQual'].value_counts()


# In[282]:

train['TotRmsAbvGrd'].value_counts().plot(kind='bar')
train['TotRmsAbvGrd'].value_counts()


# In[284]:

train['Functional'].value_counts().plot(kind='bar')
train['Functional'].value_counts()


# In[285]:

train['Fireplaces'].value_counts().plot(kind='bar')
train['Fireplaces'].value_counts()


# In[286]:

train['FireplaceQu'].value_counts().plot(kind='bar')
train['FireplaceQu'].value_counts()


# In[288]:

train['GarageType'].value_counts().plot(kind='bar')
train['GarageType'].value_counts()


# In[290]:

train['GarageYrBlt'].hist()


# In[291]:

train['GarageFinish'].value_counts().plot(kind='bar')
train['GarageFinish'].value_counts()


# In[292]:

train['GarageCars'].value_counts().plot(kind='bar')
train['GarageCars'].value_counts()


# In[294]:

train['GarageArea'].hist()


# In[295]:

train['GarageQual'].value_counts().plot(kind='bar')
train['GarageQual'].value_counts()


# In[296]:

train['GarageCond'].value_counts().plot(kind='bar')
train['GarageCond'].value_counts()


# In[297]:

train['PavedDrive'].value_counts().plot(kind='bar')
train['PavedDrive'].value_counts()


# In[299]:

train['WoodDeckSF'].hist()


# In[300]:

train['OpenPorchSF'].hist()


# In[301]:

train['EnclosedPorch'].hist()


# In[302]:

train['3SsnPorch'].hist()


# In[303]:

train['ScreenPorch'].hist()


# In[304]:

train['PoolArea'].hist()


# In[305]:

train['PoolQC'].value_counts().plot(kind='bar')
train['PoolQC'].value_counts()


# In[307]:

train['Fence'].value_counts().plot(kind='bar')
train['Fence'].value_counts()


# In[308]:

train['MiscFeature'].value_counts().plot(kind='bar')
train['MiscFeature'].value_counts()


# In[309]:

train['MiscVal'].value_counts().plot(kind='bar')
train['MiscVal'].value_counts()


# In[310]:

train['MoSold'].value_counts().plot(kind='bar')
train['MoSold'].value_counts()


# In[311]:

train['YrSold'].value_counts().plot(kind='bar')
train['YrSold'].value_counts()


# In[312]:

train['SaleType'].value_counts().plot(kind='bar')
train['SaleType'].value_counts()


# In[313]:

train['SaleCondition'].value_counts().plot(kind='bar')
train['SaleCondition'].value_counts()


# In[ ]:




# ## 2.4 Verify Data Quality
# ### Outputs:
# 
# - Data Quality Report
