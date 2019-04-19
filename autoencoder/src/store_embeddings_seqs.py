#!/usr/bin/python3
import argparse
import sys
import time
import numpy as np
import prepare_data
import prepare_similarity_matrix
import classification_module
from losses import mse_offset, mse_l1, mse_l2

from numpy import array
from keras.models import load_model
from keras.models import Model
import keras.losses

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
pathMatureAmp = '../data/train_mature.csv'
pathNonAmp = '../data/train_non_AMP.csv'
X_train, X_test, X_train_size, X_test_size = prepare_data.load_data(pathMatureAmp)
X_train_nonAMP, X_test_nonAMP, X_train_nonAMP_size, X_test_nonAMP_size = prepare_data.load_data(pathNonAmp)

# X_train_nonAMP = X_train_nonAMP[:,:174,:]
# X_test_nonAMP = X_test_nonAMP[:,:174,:]

# Define losses. 
keras.losses.mse_offset = mse_offset
keras.losses.mse_l1 = mse_l1
keras.losses.mse_l2 = mse_l2

def get_embeddings_and_sequences(model, ampSet, ampSetSizes):
    result = []
    sequences = []
    for i in range(len(ampSet)):
        # if(ampSetSizes[i]<174):
        sequence = array(ampSet[i,:,:])
        sequence = sequence.reshape((1, n_in, features))
        sequenceShort = sequence[:,:ampSetSizes[i],]
        initialSeq = prepare_data.translateOneHot(sequenceShort.reshape(ampSetSizes[i], features))
        # get the feature vector for the input sequence
        yhat = model.predict(sequence)

        rounded = np.around(yhat[0,:,], decimals=3)
        if(i == 0):
            result = [rounded]
            sequences = [initialSeq]
        else:
            result.append(rounded) 
            sequences.append(initialSeq) 
    return array(result), np.array(sequences)

def write_embeddings_and_sequences(modelname, shortModelname):
    """
    Write also sequence as first column.
    """
    model = load_model(modelname)
    modelCp = Model(inputs=model.inputs, outputs=model.layers[1].output)
    
    matureAmp = np.concatenate((X_train, X_test))
    ampSetSizes = np.concatenate((X_train_size, X_test_size))
    matureEmb, matureSeqs = get_embeddings_and_sequences(modelCp, matureAmp, ampSetSizes)    
    comb = np.column_stack((matureSeqs, matureEmb))
    np.savetxt("../embseqs/mature/{}.csv".format(shortModelname), comb, delimiter=",", fmt='%s')
    
    X_nonAMP = np.concatenate((X_train_nonAMP, X_test_nonAMP))
    X_nonAMP_sizes = np.concatenate((X_train_nonAMP_size, X_test_nonAMP_size))
    nonAmpEmb, nonAmpSeqs = get_embeddings_and_sequences(modelCp, X_nonAMP, X_nonAMP_sizes) 
    combNonAmp = np.column_stack((nonAmpSeqs, nonAmpEmb))
    np.savetxt("../embseqs/nonamp/{}.csv".format(shortModelname), combNonAmp, delimiter=",", fmt='%s')
    print("Embeddings stored in ../embseqs/..")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("modelname", action="store", help="Name of model to save.")
    args = parser.parse_args()

    modelname = "../models/"
    modelname += args.modelname 
    modelname += ".h5"
    
    write_embeddings_and_sequences(modelname, args.modelname)
