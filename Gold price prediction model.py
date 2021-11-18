#!/usr/bin/env python
# coding: utf-8

# # Importing thenecessary dependencies

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor


# # Data collection

# In[2]:


#loading the dataset
df = pd.read_csv('C:/Users/Hemant/jupyter_codes/ML Project 1/Gold price prediction/gold_price_data.csv')


# In[3]:


#to see the data in the 5 row of the dataset
df.head()


# In[4]:


#shape of the dataset
df.shape


# In[5]:


#necessary information of the dataset
df.info()


# In[6]:


#check for any null values
df.isnull().sum()


# In[7]:


#getting the statistical measure of the dataset
df.describe()


# In[8]:


#correlation between the datapoints
correlation = df.corr()


# In[9]:


#constructing a heatmap to understand the correlatin
plt.figure(figsize = (8, 8))

sns.heatmap(correlation, cbar = True, square = True, fmt = '.1f', annot = True, annot_kws = {'size': 8}, cmap ='Blues')


# In[10]:


#correlation values of GLD
correlation['GLD']


# In[11]:


#checking the distribution of gold value
sns.displot(df['GLD'], color = 'green')


# splitting the features and label

# In[12]:


X = df.drop(columns = ['Date', 'GLD'], axis =1)
Y = df['GLD']


# In[13]:


print(X)
print(Y)


# In[14]:


print(X.shape, Y.shape)


# # Splitting the dataset into training and testing data for model evaluation

# In[15]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .2, random_state = 2)


# model training :
# 
# Random forest regressor

# In[16]:


model = RandomForestRegressor(n_estimators = 100)


# In[17]:


model.fit(x_train, y_train)


# model evaluation

# In[18]:


#model evaluation on training data 
training_pred = model.predict(x_train)

# R squared error
error_score = metrics.r2_score(training_pred, y_train)

print('THE TRAINING ERROR IS :', error_score)


# In[19]:


#model evaluation on testing data 
testing_pred = model.predict(x_test)

# R squared error
error_score = metrics.r2_score(testing_pred, y_test)

print('THE TRAINING ERROR IS :', error_score)


# In[20]:


#compare the actual price and predicted price of test data
y_test = list(y_test)


# In[21]:


plt.plot(y_test, color = 'blue', label = 'ACTUAL PRICE')
plt.plot(testing_pred, color = 'green', label = 'PREDICTED PRICE')
plt.xlabel('NO. OF VALUES')
plt.ylabel('GOLD PRICE')
plt.title('ACTUAL PRICE vs PREDICTED PRICE')
plt.legend()
plt.show()


# In[ ]:




