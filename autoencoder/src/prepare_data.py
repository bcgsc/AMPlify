#!/usr/bin/python3
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
"""
This script processes an input text file to produce data in binary
format to be used with the autoencoder (binary is much faster to read).
Input data is csv file of format 'SEQUENCE,label'
"""
path = '../data/train_mature.csv'

aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
def getUniqueCharList(allSequences):
    chars = []
    for elem in allSequences:
        for ch in elem:
            if(ch not in chars):
                chars.append(ch)
    return chars

# do one hot encoding while using the features
def load_data(path):
    uniqueCharsList = []
    allSequences = []
    sizes = []
    longestSeqSize = 0

    df = pd.read_csv(path)
    
    seqsDf = df['sequence']
    labels = df['label']
    for elem in seqsDf:
        seqLen = len(elem)
        sizes.append(seqLen)
        elements = []
        for ch in elem:
            elements.append(ch)
            if(ch not in uniqueCharsList):
                uniqueCharsList.append(ch)

        allSequences.append(elements) # ([word_dict[token] for token in elements]);
        if(seqLen > longestSeqSize):
            longestSeqSize = seqLen

    # ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    features = sorted(uniqueCharsList)
    indexFeatures = {}
    for i, ch in enumerate(features):
        indexFeatures[ch] = i

    n = len(allSequences)
    d = len(features)
    X = np.zeros((n, longestSeqSize, d))
    
    # One hot encoding    
    for i in range(n):
        sequence = allSequences[i]
        for index in range(len(sequence)):
            ch = sequence[index]
            X[i, index, indexFeatures[ch]] = 1

    #test train split 
    splitFactor = int(0.8*n)
    train, test = X[:splitFactor,:], X[splitFactor:,:]
    sizesTrain, sizesTest = np.asarray(sizes[:splitFactor]), np.asarray(sizes[splitFactor:])
    
    print(train.shape, test.shape)
    print(sizesTrain.shape, sizesTest.shape)

    return train, test, sizesTrain, sizesTest

# do one hot encoding while using the features, also returns longestSeqLength
def load_data(path):
    uniqueCharsList = []
    allSequences = []
    sizes = []
    longestSeqSize = 0

    df = pd.read_csv(path)
    
    seqsDf = df['sequence']
    labels = df['label']
    for elem in seqsDf:
        seqLen = len(elem)
        sizes.append(seqLen)
        elements = []
        for ch in elem:
            elements.append(ch)
            if(ch not in uniqueCharsList):
                uniqueCharsList.append(ch)

        allSequences.append(elements) # ([word_dict[token] for token in elements]);
        if(seqLen > longestSeqSize):
            longestSeqSize = seqLen

    # ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    features = sorted(uniqueCharsList)
    indexFeatures = {}
    for i, ch in enumerate(features):
        indexFeatures[ch] = i

    n = len(allSequences)
    d = len(features)
    X = np.zeros((n, longestSeqSize, d))
    
    # One hot encoding    
    for i in range(n):
        sequence = allSequences[i]
        for index in range(len(sequence)):
            ch = sequence[index]
            X[i, index, indexFeatures[ch]] = 1

    #test train split 
    splitFactor = int(0.8*n)
    train, test = X[:splitFactor,:], X[splitFactor:,:]
    sizesTrain, sizesTest = np.asarray(sizes[:splitFactor]), np.asarray(sizes[splitFactor:])
    
    print(train.shape, test.shape)
    print(sizesTrain.shape, sizesTest.shape)

    return train, test, sizesTrain, sizesTest, longestSeqSize

# do one hot encoding while using the features
def load_data_nosplit(path):
    uniqueCharsList = []
    allSequences = []
    sizes = []
    longestSeqSize = 0

    df = pd.read_csv(path)
    
    seqsDf = df['sequence']
    labels = df['label']
    for elem in seqsDf:
        seqLen = len(elem)
        sizes.append(seqLen)
        elements = []
        for ch in elem:
            elements.append(ch)
            if(ch not in uniqueCharsList):
                uniqueCharsList.append(ch)

        allSequences.append(elements) # ([word_dict[token] for token in elements]);
        if(seqLen > longestSeqSize):
            longestSeqSize = seqLen

    # ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    features = sorted(uniqueCharsList)
    indexFeatures = {}
    for i, ch in enumerate(features):
        indexFeatures[ch] = i

    n = len(allSequences)
    d = len(features)
    X = np.zeros((n, longestSeqSize, d))
    
    # One hot encoding    
    for i in range(n):
        sequence = allSequences[i]
        for index in range(len(sequence)):
            ch = sequence[index]
            X[i, index, indexFeatures[ch]] = 1

    return X, sizes

def getSize(lower, upper):
    X_train, X_test, X_train_size, X_test_size = load_data(path)
    X_train_sized = [] # np.zeros(X_train.shape)
    X_test_sized = [] # np.zeros(X_test.shape)
    X_train_size_sized = [] # np.zeros(X_train_size.shape) 
    X_test_size_sized =  [] # np.zeros(X_test_size.shape)

    train_counter = 0
    for i in range(len(X_train_size)):
        if(X_train_size[i] >= lower and X_train_size[i] < upper):
            X_train_sized.append(X_train[i])
            X_train_size_sized.append(X_train_size[i])
            train_counter += 1
    
    test_counter = 0
    for i in range(len(X_test_size)):
        if(X_test_size[i] >= lower and X_test_size[i] < upper):
            X_test_sized.append(X_test[i])
            X_test_size_sized.append(X_test_size[i])
            test_counter += 1
    # print("New train test")
    X_train_sized = np.asarray(X_train_sized)
    X_test_sized = np.asarray(X_test_sized) # np.zeros(X_test.shape)
    X_train_size_sized = np.asarray(X_train_size_sized) # np.zeros(X_train_size.shape) 
    X_test_size_sized = np.asarray(X_test_size_sized) # np.zeros(X_test_size.shape)

    # print(X_train_sized.shape, X_test_sized.shape)
    # print(X_train_size_sized.shape, X_test_size_sized.shape)
    
    X_train_sized = np.asarray(X_train_sized)
    X_test_sized = np.asarray(X_test_sized)
    assert not np.any(np.isnan(X_train_sized))
    assert not np.any(np.isnan(X_test_sized))
    return X_train_sized, X_test_sized, np.asarray(X_train_size_sized), np.asarray(X_test_size_sized)

def getSized(lower, upper, X_train, X_test, X_train_size, X_test_size):
    # X_train, X_test, X_train_size, X_test_size = load_data(path)
    X_train_sized = [] # np.zeros(X_train.shape)
    X_test_sized = [] # np.zeros(X_test.shape)
    X_train_size_sized = [] # np.zeros(X_train_size.shape) 
    X_test_size_sized =  [] # np.zeros(X_test_size.shape)

    train_counter = 0
    for i in range(len(X_train_size)):
        if(X_train_size[i] >= lower and X_train_size[i] < upper):
            X_train_sized.append(X_train[i])
            X_train_size_sized.append(X_train_size[i])
            train_counter += 1
    
    test_counter = 0
    for i in range(len(X_test_size)):
        if(X_test_size[i] >= lower and X_test_size[i] < upper):
            X_test_sized.append(X_test[i])
            X_test_size_sized.append(X_test_size[i])
            test_counter += 1
    # print("New train test")
    X_train_sized = np.asarray(X_train_sized)
    X_test_sized = np.asarray(X_test_sized) # np.zeros(X_test.shape)
    X_train_size_sized = np.asarray(X_train_size_sized) # np.zeros(X_train_size.shape) 
    X_test_size_sized = np.asarray(X_test_size_sized) # np.zeros(X_test_size.shape)

    # print(X_train_sized.shape, X_test_sized.shape)
    # print(X_train_size_sized.shape, X_test_size_sized.shape)
    
    X_train_sized = np.asarray(X_train_sized)
    X_test_sized = np.asarray(X_test_sized)
    assert not np.any(np.isnan(X_train_sized))
    assert not np.any(np.isnan(X_test_sized))
    return X_train_sized, X_test_sized, np.asarray(X_train_size_sized), np.asarray(X_test_size_sized)
    
def translateOneHot(matrix):
    length, aminoAcids = matrix.shape
    # print(length, aminoAcids)
    seq = ""
    for row in matrix:
        indecesOfOne = np.argmax(row)
        character = aa[indecesOfOne]
        seq += character
    return seq

def translateMaxOneHot(matrix):
    from keras import backend as K 
    seq = ""
    if(K.is_tensor(matrix) != True):
        length, aminoAcids = matrix.shape
        for row in matrix:
            if(row.all() != 0):
                indecesOfOne = np.argmax(row)
                character = aa[indecesOfOne]
                seq += character
            else:
                break # once we reach this we are done? 
    return seq

def hamming(seq1, seq2):
    """
    returns the hamming distance between two 
    equal strings, and distance
    """
    res = 0
    for i in range(len(seq1)):
        if(seq1[i] != seq2[i]):
            res += 1
    return res, len(seq1)

# X_train, X_test, X_train_size, X_test_size = load_data('../data/train_mature.csv')
# X_train, X_test, X_train_size, X_test_size = getSized(0, 40, X_train, X_test, X_train_size, X_test_size)
# print(X_train.shape, X_test.shape)