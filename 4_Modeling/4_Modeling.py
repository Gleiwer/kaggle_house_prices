
# coding: utf-8

# # 4. Modeling
# ## 4.1 Select Modeling Techniques
# ### Outputs:
# - Modeling Technique
# - Modeling Assumptions
# 
# 

# In[433]:

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

from sklearn.model_selection import train_test_split, cross_val_score
import sklearn.metrics as metrics

from sklearn import tree
from sklearn.ensemble import GradientBoostingRegressor, VotingClassifier

import warnings
warnings.filterwarnings('ignore')


# In[434]:

train_dataframe= pd.read_csv("../data/train_dummied.csv")
train_dataframe=train_dataframe.set_index('Id')
y=np.array(train_dataframe.pop('SalePrice'))
X=np.array(train_dataframe)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[ ]:




# ### Logistic Regression

# In[435]:

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train, y_train)
y_pred_lr=lr.predict(X_test)


# In[436]:

print ('Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_pred_lr,y_test)))
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ###  Decision Tree Regression 

# In[437]:

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X_train, y_train)
y_pred_clf=clf.predict(X_test)


# In[438]:

print ('Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_pred_clf,y_test)))
scores = cross_val_score(clf, X_train, y_train, cv=5)
print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Ridge Linear Regression

# In[439]:

ridge = linear_model.Ridge (alpha = .5)
ridge.fit (X_train,y_train)
y_pred_ridge = ridge.predict(X_test)


# In[440]:

print ('Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_pred_ridge,y_test)))
scores = cross_val_score(ridge, X_train, y_train, cv=5)
print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Lasso Linear Regression

# In[441]:

lasso = linear_model.Lasso(alpha = 1.5)
lasso = lasso.fit(X_train,y_train)
y_pred_lasso = lasso.predict(X_test)


# In[442]:

print ('Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_pred_lasso,y_test)))
scores = cross_val_score(lasso, X_train, y_train, cv=5)
print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Elastic Net

# In[443]:

ENST = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], 
                                    l1_ratio=[.01, .1, .5, .9, .99], 
                                    max_iter=5000)
ENST = ENST.fit(X_train, y_train)
y_pred_enst = ENST.predict(X_test)


# In[444]:

print ('Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_pred_enst,y_test)))
scores = cross_val_score(ENST, X_train, y_train, cv=5)
print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Gradient Tree Boosting

# In[445]:

gtb = GradientBoostingRegressor(n_estimators=100, 
                                learning_rate=0.1,
                                max_depth=1,
                                random_state=0, 
                                loss='ls')
gtb = gtb.fit(X_train,y_train)
y_pred_gtb = gtb.predict(X_test)


# In[446]:

print ('Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_pred_gtb,y_test)))
scores = cross_val_score(gtb, X_train, y_train, cv=5)
print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Voting Classifier

# In[467]:

results=pd.DataFrame({'Ridge':y_pred_ridge,
                     'Lasso':y_pred_lasso,
                     'ENST':y_pred_enst,
                     'test':y_test})


# In[470]:

stacker= linear_model.LinearRegression()
stacker.fit(results[['Ridge', 'Lasso', 'ENST']], results['test'])


# In[471]:

scores = cross_val_score(stacker, results[['Ridge', 'Lasso', 'ENST']], results['test'], cv=5)
print("Accuracy: %0.5f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[461]:

results['stacked']=stacker.predict(results[['Ridge', 'Lasso', 'ENST']])


# In[458]:

sub_dataframe= pd.read_csv("../data/test_dummied.csv")
sub_dataframe=sub_dataframe.set_index('Id')
del sub_dataframe['SalePrice']

X_sub=np.array(sub_dataframe)

submission_dataset=pd.DataFrame({'lasso':lasso.predict(X_sub),
                                'ridge':ridge.predict(X_sub),
                                'ENST':ENST.predict(X_sub)})

predictions=stacker.predict(submission_dataset)


# In[459]:

pred_df=pd.DataFrame({'SalePrice':predictions,
                      'Id':sub_dataframe.index})
pred_df=pred_df.set_index('Id')
pred_df.to_csv('../data/submission.csv')


# In[474]:

enst_pred=pd.DataFrame({'SalePrice':ENST.predict(X_sub),
           'Id':sub_dataframe.index})
enst_pred=enst_pred.set_index('Id')
enst_pred.to_csv('../data/submission_enst.csv')


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
