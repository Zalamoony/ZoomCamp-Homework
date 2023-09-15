#!/usr/bin/env python
# coding: utf-8

# In[144]:


import pandas as pd
import numpy as np
pd.__version__


# In[86]:


house=pd.read_csv('housing.txt')


# In[87]:


house.shape


# In[88]:


house.head()


# In[89]:


house.columns


# In[90]:


house.isnull().sum()


# In[91]:


house['ocean_proximity'].nunique()


# In[92]:


#house[house['ocean_proximity']=='NEAR BAY ']['median_house_value'].mean()
house_mean=house[house['ocean_proximity']=='NEAR BAY']['median_house_value'].mean()

round(house_mean,3)


# In[93]:


round(house['total_bedrooms'].mean(),3)


# In[94]:


house['total_bedrooms'].fillna(house['total_bedrooms'].mean(),inplace=True)


# In[95]:


house['total_bedrooms'].mean()


# In[121]:


df=house[house['ocean_proximity']=='ISLAND']



# In[126]:


selected_columns=['housing_median_age','total_rooms','total_bedrooms']


# In[135]:


df=df[selected_columns]
df


# In[139]:


X=df.to_numpy()


# In[140]:


X


# In[156]:


XTX=X.T@X
XTX


# In[157]:


XTX_inv = np.linalg.inv(XTX)
XTX_inv


# In[153]:


y=np.array([950, 1300, 800, 1000, 1300])
y


# In[154]:


w = (XTX_inv @ X.T) @ y


# In[155]:


w


# In[ ]:




