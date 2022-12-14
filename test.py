#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from train import features_Glucose
import pickle
from sklearn.decomposition import PCA
import pickle_compat
pickle_compat.patch()


# In[3]:


with open("RF_Model.pkl", 'rb') as file:
        GPC_Model = pickle.load(file) 
        test_df = pd.read_csv('test.csv', header=None)
    


# In[ ]:


Features_CGM=features_Glucose(test_df)
ss_fit = StandardScaler().fit_transform(Features_CGM)


# In[ ]:


pca = PCA(n_components=5)
pca_fit=pca.fit_transform(ss_fit)


# In[ ]:


predictions = GPC_Model.predict(pca_fit)
pd.DataFrame(predictions).to_csv("Results.csv", header=None, index=False)

