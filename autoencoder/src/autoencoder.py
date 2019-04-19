#!/usr/bin/python3
import argparse
import sys
import time
import numpy as np
import prepare_data
import prepare_similarity_matrix
from losses import mse_offset, mse_l1, mse_l2

from numpy import array
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Bidirectional,LSTM
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.models import Model
from keras.layers.core import Masking
from keras.utils import plot_model
from keras.regularizers import L1L2
from keras.models import load_model
import keras.losses
from keras.callbacks import TensorBoard

import tensorflow as tf
import keras.backend as K
sess = K.get_session()

# Input specifications.
features = 20
n_in = 174
# Training specifications.
noEpochs = 200
encoding_dim = 80

aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
aad = {}
for i in range(20):
    aad[i] = aa[i]

# Load input.
X_train, X_test, X_train_size, X_test_size = prepare_data.load_data('../data/train_mature.csv')
X_train, X_test, X_train_size, X_test_size = prepare_data.getSized(0, 40, X_train, X_test, X_train_size, X_test_size)

# Define losses. 
keras.losses.mse_offset = mse_offset
keras.losses.mse_l1 = mse_l1
keras.losses.mse_l2 = mse_l2

# For the mature AMP dataset, this is an optimal separation of data.
def trainDataProvider():
    sizes = [(0,20), (20,30), (30,40),(40,160)]
    for (i,j) in sizes:
        yield prepare_data.getSized(i,j, X_train, X_test, X_train_size, X_test_size)

def autoencoder(loss):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(n_in, features)))
    model.add(LSTM(encoding_dim, activation='relu'))
    model.add(RepeatVector(n_in))
    model.add(LSTM(encoding_dim, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    model.compile(optimizer='adam', loss=loss) 
    return model

def bidir_autoencoder(loss):
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(n_in, features)))
    model.add(Bidirectional(LSTM(encoding_dim, activation='relu')))
    model.add(RepeatVector(n_in))
    model.add(LSTM(encoding_dim, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(features)))
    model.compile(optimizer='adam', loss=loss) 
    return model

def train_in_batches(modelname):
    loss = 'mse'
    model = autoencoder(loss)
    print("Training...")
    t1 = time.time()
    for e in range(noEpochs):
        print("epoch %d" % (e+1))        
        for train, test, train_size, test_size in trainDataProvider():
            model.fit(train, train, epochs=1, verbose=1)
    print("Done training in {0:.2f} seconds".format(time.time()-t1))
    
    model.save(modelname)
    del model  # deletes the existing model
    print("Model stored in {}".format(modelname))

def train(modelname, loss, savemodel=True):
    """
    Trains an autoencoder model on the specified loss.
    Added early stopping as a regularizer.
    """
    # model = autoencoder(loss)
    model = bidir_autoencoder(loss) # much better than normal LSTM 
    print("Training...")
    # TODO: For now don't use tensorboard.
    # tensorboard = TensorBoard(log_dir="../logs/", histogram_freq=0, write_graph=True, write_images=True)

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=70)
    t1 = time.time()
    model.fit(X_train, X_train, epochs=noEpochs, validation_split=0.25, verbose=2, callbacks=[es])
    print("Done training in {0:.2f} seconds".format(time.time()-t1))
    if(savemodel == False):
        return
    model.save(modelname)  # creates a HDF5 file 'my_model.h5'
    del model  # deletes the existing model
    print("Model stored in {}".format(modelname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("lowersize", action="store", type=int, help="Minimum length of amps.")
    # parser.add_argument("uppersize", action="store", type=int, help="Maximum length of amps.")
    parser.add_argument("modelname", action="store", help="Name of model to save.")
    
    args = parser.parse_args()
    modelname = "../models/"
    modelname += args.modelname 
    modelname += ".h5"
    
    loss = input("loss l1, l2 or offset? ")
    print("Using loss {}".format(loss))
    if(loss == "l1"):
        train(modelname, mse_l1)
    elif(loss == "l2"):
        train(modelname, mse_l2)
    elif(loss == "offset"):
        train(modelname, mse_offset)
    else:
        train(modelname, 'mse')