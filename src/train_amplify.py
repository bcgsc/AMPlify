#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:28:26 2019

This script is for model training, and testing the performance if a test set is specified

@author: Chenkai Li
"""

import argparse
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from layers import Attention, ScaledDotProductAttention, MultiHeadAttention
from keras.models import load_model, Model
from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


MAX_LEN = 200 # max length for input sequences


def one_hot_padding(seq_list,padding):
    """
    Generate features for aa sequences [one-hot encoding with zero padding].
    Input: seq_list: list of sequences, 
           padding: padding length, >= max sequence length.
    Output: one-hot encoding of sequences.
    """
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*20
        one_hot[aa[i]][i] = 1 
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*20]*(padding-len(seq_list[i]))
        feat_list.append(feat)   
    return(np.array(feat_list))


def build_model():
    """
    Build and compile the model.
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(name='Attention')(hidden)
    prediction = Dense(1, activation='sigmoid', name='Output')(hidden)
    model = Model(inputs=inputs, outputs=prediction)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) #best
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


def predict_by_class(scores):
    """
    Turn prediction scores into classes.
    If score > 0.5, label the sample as 1; else 0.
    Input: scores - scores predicted by the model, 1-d array.
    Output: an array of 0s and 1s.
    """
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append(1)
        else:
            classes.append(0)
    return np.array(classes)


def main():
    parser = argparse.ArgumentParser(description=dedent('''
        AMPlify training
        ------------------------------------------------------
        Given training sets with two labels: AMP and non-AMP,
        train the AMP prediction model.    
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-amp_tr', help="Training AMP set, fasta file", required=True)
    parser.add_argument('-non_amp_tr', help="Training non-AMP set, fasta file", required=True)
    parser.add_argument('-amp_te', help="Test AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-non_amp_te', help="Test non-AMP set, fasta file (optional)", default=None, required=False)
    parser.add_argument('-sample_ratio', 
                        help="Whether the training set is balanced or not (balanced by default, optional)", 
                        choices=['balanced', 'imbalanced'], default='balanced', required=False)
    parser.add_argument('-out_dir', help="Output directory", required=True)
    parser.add_argument('-model_name', help="File name of trained model weights", required=True)
    
    args = parser.parse_args()
    
    # load training sets
    AMP_train = []
    non_AMP_train = []
    for seq_record in SeqIO.parse(args.amp_tr, 'fasta'):
        # "../data/AMPlify_AMP_train_common.fa"
        AMP_train.append(str(seq_record.seq))
    for seq_record in SeqIO.parse(args.non_amp_tr, 'fasta'):
        # "../data/AMPlify_non_AMP_train_balanced.fa"
        non_AMP_train.append(str(seq_record.seq))
        
    # sequences for training sets
    train_seq = AMP_train + non_AMP_train    
    # set labels for training sequences
    y_train = np.array([1]*len(AMP_train) + [0]*len(non_AMP_train))
    
    # shuffle training set
    train = list(zip(train_seq, y_train))
    random.Random(123).shuffle(train)
    train_seq, y_train = zip(*train)
    train_seq = list(train_seq)
    y_train = np.array((y_train))
    
    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_train = one_hot_padding(train_seq, MAX_LEN) 
    
    indv_pred_train = [] # list of predicted scores for individual models on the training set
    
    # if test sets specified, process the test data
    if args.amp_te is not None and args.non_amp_te is not None:
        # load test sets
        AMP_test = []
        non_AMP_test = []      
        for seq_record in SeqIO.parse(args.amp_te, 'fasta'):
            # "../data/AMPlify_AMP_test_common.fa"
            AMP_test.append(str(seq_record.seq))
        for seq_record in SeqIO.parse(args.non_amp_te, 'fasta'):
            # "../data/AMPlify_non_AMP_test_balanced.fa"
            non_AMP_test.append(str(seq_record.seq))
        
        # sequences for test sets
        test_seq = AMP_test + non_AMP_test
        # set labels for test sequences
        y_test = np.array([1]*len(AMP_test) + [0]*len(non_AMP_test))
        # generate one-hot encoding input and pad sequences into MAX_LEN long
        X_test = one_hot_padding(test_seq, MAX_LEN)
        indv_pred_test = [] # list of predicted scores for individual models on the test set       

    ensemble_number = 5 # number of training subsets for ensemble
    ensemble = StratifiedKFold(n_splits=ensemble_number, shuffle=True, random_state=50)
    save_file_num = 0
    
    for tr_ens, te_ens in ensemble.split(X_train, y_train):
        model = build_model()
        if args.sample_ratio == 'balanced':
            early_stopping = EarlyStopping(monitor='val_acc',  min_delta=0.001, patience=50, restore_best_weights=True)
            model.fit(X_train[tr_ens], np.array(y_train[tr_ens]), epochs=1000, batch_size=32, 
                      validation_data=(X_train[te_ens], y_train[te_ens]), verbose=2, initial_epoch=0, callbacks=[early_stopping])
        else:
            model.fit(X_train[tr_ens], np.array(y_train[tr_ens]), epochs=50, batch_size=32, 
                      validation_data=(X_train[te_ens], y_train[te_ens]), verbose=2, initial_epoch=0, 
                      class_weight={0: 0.1667, 1: 0.8333})
        temp_pred_train = model.predict(X_train).flatten() # predicted scores on the [whole] training set from the current model
        indv_pred_train.append(temp_pred_train)
        save_file_num = save_file_num + 1
        save_dir = args.out_dir + '/' + args.model_name + '_' + str(save_file_num) + '.h5'
        save_dir_wt = args.out_dir + '/' + args.model_name + '_weights_' + str(save_file_num) + '.h5'
        model.save(save_dir) #save
        model.save_weights(save_dir_wt) #save

        # training and validation accuracy for the current model
        temp_pred_class_train_curr = predict_by_class(model.predict(X_train[tr_ens]).flatten())
        temp_pred_class_val = predict_by_class(model.predict(X_train[te_ens]).flatten())

        print('*************************** current model ***************************')
        print('current train acc: ', accuracy_score(y_train[tr_ens], temp_pred_class_train_curr))
        print('current val acc: ', accuracy_score(y_train[te_ens], temp_pred_class_val)) 
        
        # if test sets specified, output the test performance for the current model
        if args.amp_te is not None and args.non_amp_te is not None:
            temp_pred_test = model.predict(X_test).flatten() # predicted scores on the test set from the current model
            indv_pred_test.append(temp_pred_test)
            temp_pred_class_test = predict_by_class(temp_pred_test)
            tn_indv, fp_indv, fn_indv, tp_indv = confusion_matrix(y_test, temp_pred_class_test).ravel()
            #print(confusion_matrix(y_test, temp_pred_class_test))
            print('test acc: ', accuracy_score(y_test, temp_pred_class_test))  
            print('test sens: ', tp_indv/(tp_indv+fn_indv))
            print('test spec: ', tn_indv/(tn_indv+fp_indv))
            print('test f1: ', f1_score(y_test, temp_pred_class_test))
            print('test roc_auc: ', roc_auc_score(y_test, temp_pred_test))
            
        print('*********************************************************************')

    # if test sets specified, output the test performance for the ensemble model
    if args.amp_te is not None and args.non_amp_te is not None:
        y_pred_prob_test = np.mean(np.array(indv_pred_test), axis=0) # prediction for the test set after ensemble
        y_pred_class_test = predict_by_class(y_pred_prob_test)
                
        y_pred_prob_train = np.mean(np.array(indv_pred_train), axis=0) # prediction for the training set after ensemble
        y_pred_class_train = predict_by_class(y_pred_prob_train)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class_test).ravel()
        print('**************************** final model ****************************')
        print('overall train acc: ', accuracy_score(y_train, y_pred_class_train))
        #print(confusion_matrix(y_train, y_pred_class_train))
        print('overall test acc: ', accuracy_score(y_test, y_pred_class_test))
        print(confusion_matrix(y_test, y_pred_class_test))
        print('overall test sens: ', tp/(tp+fn))
        print('overall test spec: ', tn/(tn+fp))
        print('overall test f1: ', f1_score(y_test, y_pred_class_test))
        print('overall test roc_auc: ', roc_auc_score(y_test, y_pred_prob_test))
        print('*********************************************************************')
        
if __name__ == "__main__":
    main()
    