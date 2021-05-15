#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


data=pd.read_csv("ipl.csv")


# In[6]:


data.head(5)


# In[8]:


#--- Data cleaning---#

atr_to_drop=['mid','date','venue','batsman','bowler','striker','non-striker']
data=data.drop(atr_to_drop,axis=1)


# In[10]:


data['bat_team'].unique()


# In[11]:


consistent_team=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
       'Mumbai Indians', 'Kings XI Punjab','Royal Challengers Bangalore', 'Delhi Daredevils','Sunrisers Hyderabad']


# In[14]:


data=data[(data['bat_team'].isin(consistent_team)) & (data['bowl_team'].isin(consistent_team))]


# In[16]:


data=data[data['overs']>=5.0]


# In[17]:


data.head(5)


# In[18]:


data_encoded=pd.get_dummies(data,columns=['bat_team','bowl_team'])


# In[22]:


data_encoded.columns


# In[27]:


data_encoded=data_encoded[['bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore','bowl_team_Sunrisers Hyderabad',
                       'overs','runs', 'wickets','runs_last_5', 'wickets_last_5','total']]


# In[30]:


Y=data_encoded['total']
X=data_encoded.drop('total',axis=1)


# In[31]:


#--train test split data--#

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=.3,random_state=1234)


# In[32]:


#--train the model--#

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[38]:


data_frame=list()
data_frame=data_frame+[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,3,45,1,12,1]
df=np.array([data_frame])


# In[39]:


ans=regressor.predict(df)


# In[41]:


ans[0]


# In[42]:


#export the model#
import pickle

filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# In[ ]:




