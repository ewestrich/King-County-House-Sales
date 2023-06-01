#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st

import pickle

import numpy as np
import pandas as pd 
from numbers import Number
from scipy import stats
from geopy.distance import geodesic

import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import statsmodels.api as sm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


st.title("Kings County Housing Dashboard")

st.header("Interactive Housing Predictive Model Widget")

st.subheader("Map of Kings County, WA Housing Data")


# In[ ]:


model = pickle.load(open('daniel_pickle.pkl','rb'))
modeled_df = pickle.load(open('df_pickle.pkl','rb'))
ss = pickle.load(open('scaler.pkl','rb'))


# In[ ]:


test = modeled_df.iloc[0:1,:]


# In[ ]:


standard_test = ss.transform(test)


# In[ ]:


result = model.predict(standard_test)[0]


# In[ ]:


st.write('Common Kings County House Sold for:', result)


#     
#     st.image('house_pic.jpg')
#     
#     beds = st.selectbox('Select Number of Bedrooms:',[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])
# 
#     baths = st.selectbox('Select Number of Bathrooms',[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
#                                                   ,5.5,6.0, 6.5,7.0,7.5,8.0,8.5,9.5
#                                                   ,10.0,10.5])
#     if beds != list(test['bedrooms']):
#         new_test = test.replace({'bedrooms': list(test['bedrooms'])}, beds)
#         
# 
#     elif baths != list(test['bathrooms']):
#         new_test = test.replace({'bathrooms': list(test['bathrooms'])}, beds)
#         
#     standard_new_test = ss.transform(new_test)
#     
#     voila = model.predict(standard_new_test)[0]
#     
#     st.write('This House is valued at:', voila)
# 

# In[ ]:


st.image('house_pic.jpg')
    
new_test = test

beds = st.selectbox('Select Number of Bedrooms:',[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])

baths = st.selectbox('Select Number of Bathrooms',[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
                                                  ,5.5,6.0, 6.5,7.0,7.5,8.0,8.5,9.5
                                                  ,10.0,10.5])

new_test = test.replace({'bedrooms': list(test['bedrooms'])}, beds)
new_test = test.replace({'bathrooms': list(test['bathrooms'])}, beds)
        
standard_new_test = ss.transform(new_test)
    
voila = model.predict(standard_new_test)[0]
    
st.write('This House is valued at:', voila)

