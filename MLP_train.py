#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 16:21:53 2018

@author: cli
"""

# In[]:
# read datasets
import pandas as pd

data = pd.read_csv('/projects/btl/cli/AMP-classification/machine_learning/sampled_data_test/physicochemical_properties/train.csv')
#test = pd.read_csv('/projects/btl/cli/AMP-classification/machine_learning/sampled_data_test/physicochemical_properties/test.csv')

#data = pd.concat([train,test],ignore_index=True)

# In[]:
# one-hot encoding of amino acids
one_hot = {}
aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
for i in range(len(aa)):
    one_hot[aa[i]] = [0]*20
    one_hot[aa[i]][i] = 1 
    
for i in range(188):
    for j in range(20):
        data['%s_%s' %(i,aa[j])] = 0
        #train['%s_%s' %(i,aa[j])] = 0
        #test['%s_%s' %(i,aa[j])] = 0

for i in range(len(data)):
    for j in range(len(data.sequence[i])):
        data.loc[i,'%s_%s' %(j,data.sequence[i][j])] = 1  

# In[]:
# features: one-hot only
X = data.iloc[:,566:]
y = data.iloc[:,1]
#X_train = train.iloc[:,566:]
#y_train = train.iloc[:,1]
#X_test = test.iloc[:,566:]
#y_test = test.iloc[:,1]

# In[]:
# call the model
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(verbose = True, hidden_layer_sizes=(2000,1500,1000),early_stopping=True)
print(clf)

# In[]:
# train the model

clf.fit(X,y)

# In[]:
# save the model

from sklearn.externals import joblib

joblib.dump(clf, "/projects/btl/cli/AMP-classification/machine_learning/sampled_data_test/physicochemical_properties/MLP.pkl")

