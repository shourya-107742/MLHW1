#!/usr/bin/env python
# coding: utf-8

# <h1> Assignment </h1>

# In[55]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[85]:


train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
combine = [train_data, test_data]
apnd_data = train_data.append(test_data)


# ## Q1

# In[10]:


train_data.columns


# ## Q2

# Name,Sex,ticket,embarked,cabin

# ## Q3

# Age,Sibsp,Parch,Fare,PassengerId,Pclass

# ## Q4

# In[ ]:


Ticket


# ## Q5
# ### 1.Traindata

# In[24]:


train_data.isnull().sum()


# ### 2. Test Data

# In[25]:


test_data.isnull().sum()


# ## Q6

# In[27]:


train_data.dtypes


# ## Q7

# In[29]:


train_data.describe()


# ## Q8

# In[37]:


train_data.describe(include=["O"])


# ## Q9

# In[39]:


train_data[['Pclass', 'Survived']].groupby(['Pclass']).mean()


# here in above we can see correlation 0.62 ,so i will include this feature in predictive model

# ## Q10

# In[41]:


train_data[['Sex', 'Survived']].groupby(['Sex']).mean()


# here correlation is been higher for female(0.742038) than male(0.188908) so they have greater chance for survival

# ## Q11

# In[60]:


import seaborn as sns
histogram= sns.FacetGrid(train_data, col ='Survived', height=8)
histogram.map(plot.hist, 'Age', bins=range(0, 120, 15))
plot.xticks(range(0, 120, 15))


# 1.yes,they have high survival rate
# 2.yes,people above 80 years survived
# 3.The rate of deaths is higher in first graph from 15-30 ,so mostly they won't survive
# 4.yes,we should consider and should complete age feature for null values.
# 5.yes, they are giving sufficient info

# ## Q12

# In[68]:


histograms = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=3, aspect=2)
histograms.map(plot.hist, 'Age', bins=10)
histograms.add_legend();


# 1.Pclass =3 have most passengers but most of them aren't survived
# 2.yes, they survive 
# 3.yes, most of them survive
# 4.yes, they vary   
# 5.yes , we can consider

# ## Q13

# In[71]:


histograms = sns.FacetGrid(train_data, row='Embarked', col='Survived', height=4, aspect=2)
histograms.map(sns.barplot, 'Sex', 'Fare')
histograms.add_legend()


# 1. Yes ,higher fare people have higher chances
# 2. we should consider banding

# ## Q14

# In[72]:


train_data['Ticket'].describe()


# In[77]:


duplicate_rate = ((891-681)/891)*100
print(duplicate_rate)


# There is around 23.5% duplicate rate and there is no correlation between ticket and survival ,so yeah we can drop the ticket feature

# ## Q15

# In[82]:


apnd_data.describe(include=["O"])


# Cabin feature is incomplete here. From a total of 1309 rows combined set of train and test data only 295 rows have this feature which have null values equal to 1014.Its better to drop the Cabin feature.

# ## Q16

# In[142]:


train_data['gender']=0
train_data.loc[train_data['Sex']=='male',['gender']]=0
train_data.loc[train_data['Sex']=='female',['gender']]=1


# In[144]:


train_data.head()
#outputbelow


# ## Q17

# In[145]:


for x in range(len(train_data['Age'])):
    if np.isnan(train_data["Age"][x]) == True:
        train_data.at[x, 'Age']=np.random.uniform(train['Age'].mean(),train['Age'].std())


# In[146]:


train_data.tail()


# ## Q18

# In[147]:


print (train_data['Embarked'].value_counts())
train_data['Embarked'].fillna('S',inplace=True)


# ## Q19

# In[149]:


mode=train_data['Fare'].mode()
train_data['Fare'].fillna(mode,inplace=True)


# ## Q20

# In[131]:


train_data['FareBand']=0
train_data.loc[(train_data['Fare']>=-0.001) & (train_data['Fare']<7.91),['FareBand']]=0
train_data.loc[(train_data['Fare']>=7.91) & (train_data['Fare']<14.454),['FareBand']]=1
train_data.loc[(train_data['Fare']>=14.454) & (train_data['Fare']<31.0),['FareBand']]=2
train_data.loc[(train_data['Fare']>=31.0) & (train_data['Fare']<512.329),['FareBand']]=3


# In[133]:


train_data.isna().sum()


# In[134]:


train_data.describe()


# In[136]:


train_data.tail()


# In[ ]:




