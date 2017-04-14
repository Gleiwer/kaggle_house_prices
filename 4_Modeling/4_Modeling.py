
# coding: utf-8

# # 4. Modeling
# ## 4.1 Select Modeling Techniques
# ### Outputs:
# - Modeling Technique
# - Modeling Assumptions
# 
# 

# In[2]:

from sklearn.linear_model import LogisticRegression
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


# In[16]:

train_dataframe= pd.read_csv("../data/train_dummied.csv")
del train_dataframe['Unnamed: 0']
train_dataframe.set_index('Id')
y_train=np.array(train_dataframe['SalePrice'])
del train_dataframe['SalePrice']


# In[21]:




# ## 4.2 Generate Test Design
# ### Outputs:
# - Test Design
# 
# ## 4.3 Build Model
# ### Outputs:
# - Parameter Settings
# - Models
# - Model Descriptions
# 
# ## 4.4 Assess Model
# ### Outputs:
# 
# - Model Assessment
# - Revised Parameter
# - Settings
