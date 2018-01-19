#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:48:32 2018

@author: cli
"""
# In[]:
seq = input('Input your sequence: ')

# In[]:
import pandas as pd
data = pd.DataFrame()
data['sequence'] = [seq]

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

for i in range(len(data)):
    for j in range(len(data.sequence[i])):
        data.loc[i,'%s_%s' %(j,data.sequence[i][j])] = 1
        
# In[]:
X = data.iloc[:,1:]

# In[]:
from sklearn.externals import joblib
clf = joblib.load('/projects/btl/cli/AMP-classification/machine_learning/sampled_data_test/physicochemical_properties/proposal/MLP.pkl')

# In[]:
pred_proba = clf.predict_proba(X)
pred = clf.predict(X)


# In[]:
print('Your peptide: ' + seq )
print('Your peptide is ' + pred[0])
print('Confidence scores for your peptide to be AMP_mature, AMP_precursor and non_AMP: ' + str(pred_proba[0,0]) + ', ' + str(pred_proba[0,1]) + ', ' + str(pred_proba[0,2]))    