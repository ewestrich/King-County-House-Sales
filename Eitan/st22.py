#!/usr/bin/env python
# coding: utf-8

# ## Final Project Submission
# 
# Please fill out:
# * Student name: 
# * Student pace: self paced / part time / full time
# * Scheduled project review date/time: 
# * Instructor name: 
# * Blog post URL:
# 

# In[163]:


import streamlit as st


# In[164]:


import streamlit as st

#st.write("Hello ,let's learn how to build a streamlit app together")

import pandas as pd
import numpy as np
import streamlit as st
#df = pd.DataFrame(np.random.randn(500, 2) / [50, 50] + [37.76, -122.4],columns=['lat', 'lon'])
#st.map(df)

st.title ("Kings County Housing Dashboard")

st.header("Interactive Housing Predictive Model Widget")
#st.markdown("this is the header")
#st.subheader("this is the subheader")
#st.caption("this is the caption")
#st.code("x=2021")
#st.latex(r''' a+a r^1+a r^2+a r^3 ''')

#st.checkbox('yes')

#st.button('Click')

#st.radio('Pick your gender',['Male','Female'])

#st.selectbox('Pick your gender',['Male','Female'])

#st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])

#st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])

#st.slider('Pick a number', 0,50)



# In[ ]:





# In[165]:


import itertools
import numpy as np
import pandas as pd 
from numbers import Number
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from geopy.distance import geodesic


# In[166]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


# In[167]:


data = pd.read_csv('../data/kc_house_data.csv')


# In[168]:


data = data.dropna()


# In[169]:


def zip_code(address):
    x = address.split(' ')[-3]

    return x.split(',')[0]


# In[170]:


def township(address):
    x = address.split(', ')[1]

    return x


# In[171]:


data['zip_code'] = data['address'].apply(lambda x: zip_code(x))


# In[172]:


data = data.sort_values(by = 'zip_code')


# In[173]:


data['zip_code'] = data['zip_code'].astype('int')


# In[174]:


data['township'] = data['address'].apply(lambda x: township(x))


# In[182]:





# In[87]:


kc_zips =[98001,
98002,
98003,
98004,
98005,
98006,
98007,
98008,
98009,
98010,
98011,
98013,
98014,
98015,
98019,
98022,
98023,
98024,
98025,
98027,
98028,
98029,
98030,
98031,
98032,
98033,
98034,
98035,
98038,
98039,
98040,
98041,
98042,
98045,
98047,
98050,
98051,
98052,
98053,
98054,
98055,
98056,
98057,
98058,
98059,
98062,
98063,
98064,
98065,
98070,
98071,
98072,
98073,
98074,
98075,
98077,
98083,
98089,
98092,
98093,
98101,
98102,
98103,
98104,
98105,
98106,
98107,
98108,
98109,
98111,
98112,
98113,
98114,
98115,
98116,
98117,
98118,
98119,
98121,
98122,
98124,
98125,
98126,
98127,
98129,
98131,
98132,
98133,
98134,
98136,
98138,
98139,
98141,
98144,
98145,
98146,
98148,
98151,
98154,
98155,
98158,
98160,
98161,
98164,
98165,
98166,
98168,
98170,
98171,
98174,
98175,
98177,
98178,
98181,
98184,
98185,
98188,
98190,
98191,
98194,
98195,
98198,
98199,
98224,
98288]


# In[188]:


data_kc = data[data['zip_code'].isin(kc_zips)]


# In[190]:


data_kc
coords = data_kc[['lat', 'long']].rename(columns = {'long':'lon'})


# In[191]:


st.map(coords)


# In[ ]:





# In[ ]:





# In[89]:


def geo_distance(coord_a, coord_b):
    #take two coordinates and calculate distances in miles
    
    return geodesic(coord_a, coord_b).miles


# In[90]:


amazing_coord = (47.615722, -122.339494)


# In[91]:


data_kc['location'] = list(zip(data_kc.lat, data_kc.long))


# In[92]:


data_kc['distance_to_amazon'] = data_kc['location'].apply(lambda x: geodesic(x, amazing_coord).miles)


# In[93]:


data_near_amzn = data_kc[data_kc['distance_to_amazon'] <= 3]


# In[94]:


data_near_amzn_filt = data_near_amzn[data_near_amzn['price'] < 10000000]


# In[95]:


data_near_amzn_filt.columns


# In[96]:


fig, ax = plt.subplots(figsize = (12,8))

sqft = data_near_amzn_filt['sqft_living']
price = data_near_amzn_filt['price']
bedrooms = data_near_amzn_filt['bedrooms']
nuisance = data_near_amzn_filt['nuisance']
dist = data_near_amzn_filt['distance_to_amazon']
greenbelt = data_near_amzn_filt['greenbelt']
bathrooms = data_near_amzn_filt['bathrooms']


sns.scatterplot(x = sqft, y = price, hue = bedrooms, hue_norm=(0, 5), size = bathrooms, ax=ax, sizes=(10, 150), legend="full");



# In[ ]:





# In[97]:


fig, ax = plt.subplots(figsize=(12,10)) 
ax = sns.heatmap(data_kc[['price', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'sqft_above',
       'sqft_basement', 'sqft_garage', 'sqft_patio', 'floors', 'waterfront', 'greenbelt', 'nuisance', 'view',
       'condition', 'grade', 'heat_source', 'sewer_system', 'yr_built',
       'yr_renovated', 'zip_code']].corr(), annot=True, linewidths=.3)


# In[ ]:





# In[98]:


#X = data_kc[['sqft_lot' ,'sqft_living', 'floors']]
#y = data_kc['price']


# In[99]:


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[100]:


#ss = StandardScaler()


# In[101]:


#ss.fit(X_train)


# In[102]:


#X_standardized_train = ss.transform(X_train)


# In[103]:


#X_standardized_test = ss.transform(X_test)


# In[104]:


#lr_raw = LinearRegression()
#lr_raw.fit(X_standardized_train, y_train)

#lr_raw.score(X_standardized_train, y_train)


# In[105]:


#lr_raw.score(X_standardized_test, y_test)


# In[106]:


#pd.Series(lr_raw.coef_,
#          index = X.columns) 


# In[107]:


#y_pred = lr_raw.predict(X_standardized_test)
#mean_absolute_error(y_pred, y_test)


# In[108]:


#new_kcdf = pd.concat([data_kc.reset_index().drop(columns = ['grade', 'condition', 'view']), ord_df],1)


# In[ ]:





# In[109]:


#o_enc = OrdinalEncoder(categories = [grade_list, cond_list, view_list])
#ord_df =pd.DataFrame(o_enc.fit_transform(closekcdf[['grade', 'condition', 'view']]), columns = closekcdf[['grade', 'condition', 'view']].columns)


# In[ ]:





# In[110]:


#onehot_enc = OneHotEncoder(sparse = True, handle_unknown = 'ignore')


# In[111]:


#nominal_cols = ['waterfront', 'greenbelt', 'heat_source', 'sewer_system','zip_code']


# In[112]:


#ohe_df =onehot_enc.fit_transform(new_kcdf[nominal_cols])


# In[113]:


#nominal_df = pd.DataFrame(ohe_df.toarray(),columns = onehot_enc.get_feature_names())


# In[114]:


#cleandf = pd.concat([new_kcdf.drop(columns = ['waterfront', 'greenbelt', 'heat_source', 'sewer_system', 'zip_code']), nominal_df],1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[115]:


amazing_coord = (47.615722, -122.339494)


# In[116]:


#data['location'] = list(zip(data.lat, data_focused.long))


# In[117]:


#data_focused['distance_to_amazon'] = data_focused['location'].apply(lambda x: geodesic(x, amazing_coord).miles)


# In[118]:


def geo_distance(coord_a, coord_b):
    #take two coordinates and calculate distances in miles
    
    return geodesic(coord_a, coord_b).miles


# In[ ]:





# In[119]:


#X_train, X_test, y_train, y_test = train_test_split(data_focused['grade_num'], data_focused['price'], test_size = 0.3)

#lr_simple = LinearRegression()


# In[120]:


#lr_simple.fit(X_train.values.reshape(-1,1), y_train)


# In[121]:


#lr_simple.score(X_train.values.reshape(-1,1),
              #  y_train)


# In[122]:


#y_pred = lr_simple.predict(X_test.values.reshape(-1,1))
#mean_absolute_error(y_pred, y_test)


# In[123]:


data_kc = data_kc.drop(columns = ['township',  'date'])


# ### MODELING

# In[124]:


grade_list = ['2 Substandard','3 Poor','4 Low','5 Fair','6 Low Average', '7 Average', '8 Good', '9 Better', '10 Very Good', '11 Excellent','12 Luxury','13 Mansion']
cond_list = ['Poor', 'Fair', 'Average', 'Good', 'Very Good']
view_list = ['NONE', 'FAIR', 'AVERAGE', 'GOOD', 'EXCELLENT']


# In[125]:


o_enc = OrdinalEncoder(categories = [grade_list, cond_list, view_list])
ord_df =pd.DataFrame(o_enc.fit_transform(data_kc[
    ['grade', 'condition', 'view']]), columns = data_kc[['grade', 'condition', 'view']].columns)


# In[126]:


new_kcdf = pd.concat([data_kc.reset_index().drop(columns = ['grade', 'condition', 'view']), ord_df],1)


# In[127]:


data_kc


# In[128]:


onehot_enc = OneHotEncoder(sparse = True, handle_unknown = 'ignore')


# In[129]:


nominal_cols = ['waterfront', 'greenbelt', 'heat_source', 'sewer_system','zip_code', 'nuisance']


# In[130]:


ohe_df =onehot_enc.fit_transform(new_kcdf[nominal_cols])


# In[131]:


nominal_df = pd.DataFrame(ohe_df.toarray(),columns = onehot_enc.get_feature_names_out())


# In[132]:


cleandf = pd.concat([new_kcdf.drop(columns = 
    ['waterfront', 'greenbelt', 'heat_source', 'sewer_system', 'zip_code', 'nuisance']), nominal_df],1)


# In[133]:


lr = LinearRegression()


# In[134]:


col_select = cleandf.drop(columns = ['price', 'index', 'address','sqft_garage', 'location']).columns
X = cleandf[col_select]
y = cleandf['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[135]:


lr.fit(X_train, y_train)


# In[136]:


lr.score(X_train, y_train)


# In[137]:


y_pred = lr.predict(X_test)
y_pred[0:10]


# In[138]:


MAE = mean_absolute_error(y_pred, y_test)
MAE


# In[139]:


ss = StandardScaler()


# In[140]:


ss.fit(X_train)


# In[141]:


X_standardized_train = ss.fit_transform(X_train)


# In[142]:


X_standardized_test = ss.transform(X_test)


# In[143]:


lr.fit(X_standardized_train, y_train)

lr.score(X_standardized_train, y_train)


# In[144]:


lr.score(X_standardized_test, y_test)


# In[145]:


mean_absolute_error(y_pred, y_test)


#  results.summary()

# In[146]:


ss_preds = lr.predict(X_standardized_test)


# In[147]:


ss_preds


# In[148]:


#model_predict_rounded = [round(model_full_predict[x], 1) for x in range(len(model_full_predict))]


# In[149]:


#data_kc['model_predict'] = model_predict_rounded


# In[150]:


#data_kc['mean_sq_er'] = (data_kc['price'] - data_kc['model_predict'])**2

