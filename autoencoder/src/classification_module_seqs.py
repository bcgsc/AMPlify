#!/usr/bin/python3
## Importing required Libraries
import argparse
import sys
import time
import numpy as np
import prepare_data
import prepare_similarity_matrix
import os
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import pandas as pd 

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

# TODO: Data source fix.
train = pd.read_csv('../data/classification/train_20181213.csv')
test = pd.read_csv('../data/classification/test_20181213.csv')
# put train and test set into hashtable: {seq:label}
trainHash = {}
for index, row in train.iterrows():
    trainHash[row['sequence']] = row['label']
testHash = {}
for index, row in test.iterrows():
    testHash[row['sequence']] = row['label']

def MLPModel():
    return MLPClassifier(hidden_layer_sizes=(2000,1500,1000),early_stopping=True)

def grid_sarch():
    mlp = MLPClassifier(max_iter=100, early_stopping=True)
    parameter_space = { 'hidden_layer_sizes': [(200,150,100), (300,300), (100,)],
                        'activation': ['tanh', 'relu'],
                        'solver': ['sgd', 'adam'],
                        'alpha': [0.0001, 0.001, 0.01, 0.05],
                        'learning_rate': ['constant','adaptive'] }
    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
    return clf


if __name__ == '__main__':
    """
    Used for comparing models with each other. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("modelname", action="store", help="Name of model to save.")
    args = parser.parse_args()

    modelname = "../models/"
    modelname += args.modelname 
    modelname += ".h5"
    
    matureDataPath = "../embseqs/mature/{}.csv".format(args.modelname)
    nonAmpDataPath = "../embseqs/nonamp/{}.csv".format(args.modelname)
    dataMature = pd.read_csv(matureDataPath,  header=None) 
    dataNonAmp = pd.read_csv(nonAmpDataPath,  header=None) 
    
    seqsMature = np.array(dataMature.iloc[:,0:1])
    seqsNonAMP = np.array(dataNonAmp.iloc[:,0:1])
    
    embeddings = np.array(dataMature.iloc[:,1:,])
    embeddingsNonAMP = np.array(dataNonAmp.iloc[:,1:,])
    
    # get embedding for sequences in trainHash and testHash
    trainInstances = len(trainHash)
    testInstances = len(testHash)
    
    X_train = []
    X_train_sequences = []
    X_test = []
    X_test_sequences = []
    y_train = []
    y_test = []

    for i  in range(len(seqsMature)):
        if(seqsMature[i][0] in trainHash):
            X_train.append(embeddings[i])
            X_train_sequences.append(seqsMature[i][0])
            if(trainHash[seqsMature[i][0]] == 'AMP'):
                y_train.append(1)
            else:
                y_train.append(0)
        elif(seqsMature[i][0] in testHash):
            X_test.append(embeddings[i])
            X_test_sequences.append(seqsMature[i][0])
            if(testHash[seqsMature[i][0]] == 'AMP'):
                y_test.append(1)
            else:
                y_test.append(0)
                
    for i  in range(len(seqsNonAMP)):
        if(seqsNonAMP[i][0] in trainHash):
            X_train.append(embeddingsNonAMP[i])
            X_train_sequences.append(seqsNonAMP[i][0])
            if(trainHash[seqsNonAMP[i][0]] == 'AMP'):
                y_train.append(1)
            else:
                y_train.append(0)
        elif(seqsNonAMP[i][0] in testHash):
            X_test.append(embeddingsNonAMP[i])
            X_test_sequences.append(seqsNonAMP[i][0])
            if(testHash[seqsNonAMP[i][0]] == 'AMP'):
                y_test.append(1)
            else:
                y_test.append(0)
    
    print(len(X_test), len(X_test_sequences), len(y_test), testInstances)
    print(len(X_train), len(X_train_sequences), len(y_train), trainInstances)
    print("missed {} train instances and {} test instances".format(trainInstances-len(X_train), testInstances-len(X_test)))
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # model = MLPModel().fit(X_train, y_train)
    # print("Train accuracy: {}".format(model.score(X_train, y_train)))
    # print("Test accuracy: {}".format(model.score(X_test,y_test)))
    clf = grid_sarch().fit(X_train,y_train)
    # Best paramete set
    print('Best parameters found:\n', clf.best_params_)

    # All results
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    # prediction on test set
    y_true, y_pred = y_test , clf.predict(X_test)

    print('Results on the test set:')
    print(classification_report(y_true, y_pred))

    # TODO: add grid search.
    '''
    # store each sequence prediction.
    f = open("../results/train/{}.csv".format(args.modelname), 'w+')
    ft = open("../results/test/{}.csv".format(args.modelname), 'w+')
    
    print("Storing sequence predictions.")
    for i in range(len(X_train)):
        if(trainHash[X_train_sequences[i]] == 'AMP'):
            f.write("{},1,{}\n".format(X_train_sequences[i],model.predict(X_train[i].reshape(1, -1))[0]))
        else:
            f.write("{},0,{}\n".format(X_train_sequences[i], model.predict(X_train[i].reshape(1, -1))[0]))
        
    for i in range(len(X_test)):
        if(testHash[X_test_sequences[i]] == 'AMP'):
            ft.write("{},1,{}\n".format(X_test_sequences[i], model.predict(X_test[i].reshape(1, -1))[0]))
        else:
            ft.write("{},0,{}\n".format(X_test_sequences[i], model.predict(X_test[i].reshape(1, -1))[0]))
    print("Done. Predictions are in ../results/")
    '''