#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 12:33:53 2017

@author: cli
"""

# In[2]:

import pandas as pd
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#print(train)
#print(test)


# In[ ]:
# one-hot vectors
one_hot = {}
aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
for i in range(len(aa)):
    one_hot[aa[i]] = [0]*20
    one_hot[aa[i]][i] = 1 
    
for i in range(188):
    for j in range(20):
        train['%s_%s' %(i,aa[j])] = 0
        test['%s_%s' %(i,aa[j])] = 0

for i in range(len(train)):
    for j in range(len(train.sequence[i])):
        train.loc[i,'%s_%s' %(j,train.sequence[i][j])] = 1  

for i in range(len(test)):
    for j in range(len(test.sequence[i])):
        test.loc[i,'%s_%s' %(j,test.sequence[i][j])] = 1  
    

print(train)
print(test)

# In[9]:

from sklearn.model_selection import StratifiedKFold
#X_train = train.iloc[:,2:]
#y_train = train.iloc[:,1]
#X_test = test.iloc[:,2:]
#y_test = test.iloc[:,1]

# one-hot only
X_train = train.iloc[:,566:]
y_train = train.iloc[:,1]
X_test = test.iloc[:,566:]
y_test = test.iloc[:,1]



# In[ ]:
#import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(2000, activation='relu', input_dim=188*20))
model.add(Dropout(0.6))
model.add(Dense(1500, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

#model.fit(x_train, y_train,
#          epochs=20,
#          batch_size=128)

# In[ ]:
# transform targets into 3-d vectores
#from keras.utils.np_utils import to_categorical
#y_binary = to_categorical(y_int)
import numpy as np
y_train_c = np.zeros((len(y_train),3))
y_test_c = np.zeros((len(y_test),3))
for i in range(len(y_train)):
    if y_train[i] == 'AMP_mature':
        y_train_c[i][0]=1
    if y_train[i] == 'AMP_precursor':
        y_train_c[i][1]=1
    if y_train[i] == 'non_AMP':
        y_train_c[i][2]=1
for i in range(len(y_test)):
    if y_test[i] == 'AMP_mature':
        y_test_c[i][0]=1
    if y_test[i] == 'AMP_precursor':
        y_test_c[i][1]=1
    if y_test[i] == 'non_AMP':
        y_test_c[i][2]=1



# In[41]:
from keras.callbacks import EarlyStopping
X_train = np.array(X_train)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train, y_train_c, batch_size=30,validation_split=0.2,verbose=1,epochs=50, initial_epoch=0, callbacks=[early_stopping])
#from sklearn.model_selection import cross_val_score
#from sklearn import cross_validation
#kfold = StratifiedKFold(n_splits=5, shuffle=True)
#scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'accuracy')

# In[]:
# predict 
y_predict = model.predict(np.array(X_test))
print(y_predict)



