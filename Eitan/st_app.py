#!/usr/bin/env python
# coding: utf-8

# In[162]:


import streamlit as st

import pickle

import numpy as np
import pandas as pd 
from numbers import Number
from scipy import stats

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


# In[163]:


st.title("Kings County Housing Dashboard")

st.header("Interactive Housing Predictive Model Widget")

st.subheader("Map of Kings County, WA Housing Data")


# In[212]:


df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))
ss = pickle.load(open('scaler.pkl','rb'))


# In[198]:


list(df.loc[:5].columns)


# In[254]:


house = df.loc[:0, :]


# In[221]:


col = df.columns


# In[229]:


df


# In[243]:


row['sqft_living'] = 1990.0


# In[244]:


row


# In[222]:


row = df.loc[5]


# In[245]:


test_2 = pd.DataFrame(row, col).T


# In[246]:


stand_2 = ss.transform(test_2)


# In[247]:


model.predict(stand_2)


# In[211]:


test


# In[208]:


a = stand.reshape(1,-1)


# In[210]:


stand = ss.transform(test)


# In[209]:


model.predict(a)


# In[188]:


mode_house = modeled_df.iloc[0:1,:]


# In[123]:


mode_house.loc[:,'yr_built'] = 2021


# In[125]:


standard_mode_house = ss.transform(mode_house)


# In[126]:


result_mode_house = model.predict(standard_mode_house)[0]


# In[127]:


result_mode_house


# In[40]:


st.write('Median Price for Kings County:', '$869,000')


# In[128]:


kc_zips = [98001, 98002,98003, 98004,98005,98006, 98007,98008,98009, 98010, 98011, 98013, 98014,98015,98019,98022,98023,98024,98025,98027,98028,98029,98030,98031,98032,98033,98034,98035,98038,98039,98040,98041,98042,98045,98047,98050,98051,98052,98053,98054,98055,98056,98057,98058,98059,98062,98063,98064,98065,98070,98071,98072,98073,98074,98075,98077,98083,98089,98092,98093,98101,98102,98103,98104,98105,98106,98107,98108,98109,98111,98112,98113,98114,98115,98116,98117,98118,98119,98121,98122,98124,98125,98126,98127,98129,98131,98132,98133,98134,98136,98138,98139,98141,98144,98145,98146,98148,98151,98154,98155,98158,98160,98161,98164,98165,98166,98168,98170,98171,98174,98175,98177,98178,98181,98184,98185,98188,98190,98191,98194,98195,98198,98199,98224,98288]


# In[255]:


house


# In[270]:


new_house['zip_code_98007']


# new_house.columns[-43]

# new_house

# In[299]:


list(house.columns)


# In[301]:


st.image('house_pic.jpg')
    
new_house


beds = st.selectbox('Select Number of Bedrooms',[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13])

baths = st.selectbox('Select Number of Bathrooms',[0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0
                                                  ,5.5,6.0, 6.5,7.0,7.5,8.0,8.5,9.5
                                                  ,10.0,10.5])

zip_code = st.selectbox('Select a Zip Code', kc_zips)
zip_col = 'zip_code_' + str(zip_code)

sqft_living = st.number_input('Input Square Feet of Living Space')


new_house = new_house.replace({'bedrooms': list(new_house['bedrooms'])}, beds)
new_house = new_house.replace({'bathrooms': list(new_house['bathrooms'])}, baths)
new_house = new_house.replace({zip_col: list(new_house[zip_col])}, 1.0)
new_house = new_house.replace({'sqft_living': list(new_house['sqft_living'])}, sqft_living)



st.number_input

st.text_input


standard_new_house = ss.transform(new_house)
    
result_new_house = model.predict(standard_new_house)[0]

if st.button('View House Sale Price'):
    st.write('This House is valued at:', result_new_house)


# In[305]:


og_house = new_house


# In[307]:


new_house


# beds = 7
# baths = 5
# zip_col = 98198
# zip_col = 'zip_code_' + str(zip_code)
# sqft_living = 1180
# 
# new_house = new_house.replace({'bedrooms': list(new_house['bedrooms'])}, beds)
# new_house = new_house.replace({'bathrooms': list(new_house['bathrooms'])}, baths)
# new_house = new_house.replace({zip_col: list(new_house[zip_col])}, 1.0)
# new_house = new_house.replace({'sqft_living': list(new_house['sqft_living'])}, sqft_living)
# 

# standard_new_house = ss.transform(new_house)
#     
# result_new_house = model.predict(standard_new_house)[0]

# result_new_house

# In[ ]:





# standard_new_house = ss.transform(test)
#     
# result_new_house = model.predict(standard_new_house)[0]
# result_new_house

# In[ ]:




