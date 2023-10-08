#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[2]:


data=pd.read_csv(r'C:\Users\ASUS\Downloads\smoke_detection_iot.csv')


# In[3]:


data


# In[4]:


data.describe()


# In[5]:


data.isnull().sum()


# In[6]:


data.corr()['Fire Alarm']


# In[7]:


data.drop(['eCO2[ppm]', 'Raw H2', 'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5'], axis=1, inplace=True)


# In[8]:


data.columns


# In[9]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables=data[[
    'Unnamed: 0', 
#     'UTC', 
    'Temperature[C]', 
#     'Humidity[%]', 
    'TVOC[ppb]',
#     'Raw Ethanol', 
#     'Pressure[hPa]', 
    'CNT'
]]
vif=pd.DataFrame()
vif['VIF']=[variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['Features']=variables.columns

vif


# In[10]:


data.columns


# In[11]:


data.drop(['UTC', 'Humidity[%]', 'Raw Ethanol', 'Pressure[hPa]'], axis=1, inplace=True)


# In[12]:


data.head()


# In[13]:


data.drop('Unnamed: 0', axis=1, inplace=True)


# In[14]:


data.head()


# In[15]:


for i in data[['Temperature[C]', 'TVOC[ppb]', 'CNT']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[16]:


q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR=q3-q1
Lower=q1-1.5*IQR
Upper=q3+1.5*IQR


# In[17]:


for i in data[['Temperature[C]', 'TVOC[ppb]', 'CNT']]:
    data[i] = np.where(data[i] > Upper[i], Upper[i],data[i])
    data[i] = np.where(data[i] < Lower[i], Lower[i],data[i]) 


# In[18]:


for i in data[['Temperature[C]', 'TVOC[ppb]', 'CNT']]:
    sns.boxplot(x=data[i], data=data)
    plt.show()


# In[19]:


data=data.reset_index(drop=True)


# In[20]:


data.head()


# In[21]:


X=data.drop('Fire Alarm', axis=1)
y=data['Fire Alarm']


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score


# In[24]:


def evaluate(model, X_test, y_test):
    
    y_pred_test = model.predict(X_test)
    y_prob_test = model.predict_proba(X_test)[:,1]
    
    roc_score_test = roc_auc_score(y_test, y_prob_test)
    gini_score_test= roc_score_test*2-1
    
    
    y_pred_train = model.predict(X_train)
    y_prob_train = model.predict_proba(X_train)[:,1]
    
    roc_score_train = roc_auc_score(y_train, y_prob_train)
    gini_score_train = roc_score_train*2-1
    

    print('Model Performance')

    print('Gini Score for Test:', gini_score_test*100)
    print('Gini Score for Train:', gini_score_train*100)


# In[25]:


base_model = SVC(probability=True)
base_model.fit(X_train, y_train)


# In[26]:


Accuracy_basic=evaluate(base_model, X_test, y_test)


# In[27]:


from sklearn.model_selection import RandomizedSearchCV

kernel = ['linear', 'poly', 'rbf', 'sigmoid']

gamma = ['scale', 'auto'] 

C = [1, 10, 100, 1000]


random_grid = {'kernel': kernel,
               'gamma': gamma,
               'C': C}
print(random_grid)


# In[ ]:


svr_randomized = RandomizedSearchCV(estimator = base_model, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=1, n_jobs = -1)

svr_randomized.fit(X_train, y_train)


# In[ ]:


svr_randomized.best_params_


# In[ ]:


best_model=svr_randomized.best_estimator_
Accuracy_best=evaluate(best_model, X_test, y_test)

