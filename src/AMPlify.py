#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:58:13 2019

This script is for generating prediction results for test sequences

@author: Chenkai Li
"""


import os
import argparse
from textwrap import dedent
import time
from Bio import SeqIO
import numpy as np
import pandas as pd
from layers import Attention, MultiHeadAttention
from keras.models import Model
from keras.layers import Masking, Dense, LSTM, Bidirectional, Input, Dropout


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


def build_amplify():
    """
    Build the complete model architecture
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
    return model


def build_attention():
    """
    Build the model architecture for attention output
    """
    inputs = Input(shape=(MAX_LEN, 20), name='Input')
    masking = Masking(mask_value=0.0, input_shape=(MAX_LEN, 20), name='Masking')(inputs)
    hidden = Bidirectional(LSTM(512, use_bias=True, dropout=0.5, return_sequences=True), name='Bidirectional-LSTM')(masking)
    hidden = MultiHeadAttention(head_num=32, activation='relu', use_bias=True, 
                                return_multi_attention=False, name='Multi-Head-Attention')(hidden)
    hidden = Dropout(0.2, name = 'Dropout_1')(hidden)
    hidden = Attention(return_attention=True, name='Attention')(hidden)
    model = Model(inputs=inputs, outputs=hidden)
    return model


def load_multi_model(model_dir_list, architecture):
    """
    Load multiple models with the same architecture in one function.
    Input: list of saved model weights files.
    Output: list of loaded models.
    """
    model_list = []
    for i in range(len(model_dir_list)):
        model = architecture()
        model.load_weights(model_dir_list[i], by_name=True)
        model_list.append(model)
    return model_list


def ensemble(model_list, X):
    """
    Ensemble the list of models with processed input X, 
    Return results for ensemble and individual models
    """
    indv_pred = [] # list of predictions from each individual model
    for i in range(len(model_list)):
        indv_pred.append(model_list[i].predict(X).flatten())
    ens_pred = np.mean(np.array(indv_pred), axis=0)
    return ens_pred, np.array(indv_pred)


def get_attention_scores(indv_pred_list, attention_model_list, seq_list, X):
    """
    Get attention scores of the most confident model with processed input X.
    Input: 
        inv_pred_list - list of predictions from individual models
        attention_model_list - list of attention models
        seq_list - list of peptide sequences
        X - processed input of the model
    Output: 
        attention scores for all sequences from the most confident model
        attention scores for all sequences from all models
    """
    #X = one_hot_padding(seq_list, MAX_LEN)
    # calculate all attention scores
    attention_scores_list = []
    for i in range(len(attention_model_list)):
        attention_with_padding = attention_model_list[i].predict(X)
        attention_without_padding = []
        for j in range(len(seq_list)):
            attention_without_padding.append(attention_with_padding[j][0:len(seq_list[j])].flatten())
        attention_scores_list.append(attention_without_padding)
    # choose the attention scores from the most confident model
    ens_pred = np.mean(np.array(indv_pred_list), axis=0)
    confident_attention = []
    for i in range(len(ens_pred)):
        if ens_pred[i] > 0.5:
            confident_attention.append(attention_scores_list[list(indv_pred_list[:,i]).index(max(indv_pred_list[:,i]))][i])
        else:
            confident_attention.append(attention_scores_list[list(indv_pred_list[:,i]).index(min(indv_pred_list[:,i]))][i])
    return confident_attention


def proba_to_class_name(scores):
    """
    Turn prediction scores into real class names.
    If score > 0.5, label the sample as AMP; else non-AMP.
    Input: scores - scores predicted by the model, 1-d array.
    Output: an array of class labels.
    """
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append('AMP')
        else:
            classes.append('non-AMP')
    return np.array(classes)


def main():
    parser = argparse.ArgumentParser(description=dedent('''
        AMPlify
        ------------------------------------------------------
        Predict whether a sequence is AMP or not.
        Input sequences should be in fasta format. 
        Sequences should be shorter than 201 amino acids long, 
        and should not contain amino acids other than the 20 standard ones. 
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-md', '--model_dir', help="Directory of where models are stored", required=True)
    parser.add_argument('-m', '--model_name', nargs=5, help="File names of 5 trained models (optional)", 
                        default=['model_weights_1.h5', 'model_weights_2.h5', 'model_weights_3.h5', 
                                 'model_weights_4.h5', 'model_weights_5.h5'], required=False)
    parser.add_argument('-s', '--seqs', help="Sequences for prediction, fasta file", required=True)
    parser.add_argument('-od', '--out_dir', help="Output directory (optional)", default=os.getcwd(), required=False)
    parser.add_argument('-of', '--out_format', help="Output format, txt or tsv (optional)", 
                        choices=['txt', 'tsv'], default='txt', required=False)
    parser.add_argument('-att', '--attention', help="Whether to output attention scores, on or off (optional)",
                        choices=['on', 'off'], default='off', required=False)
    
    args = parser.parse_args()

    print('\nLoading models...')
    models = [args.model_dir + '/' + args.model_name[i] for i in range(len(args.model_name))]
    # load models for final output
    out_model = load_multi_model(models, build_amplify)
    # load_models for attention
    if args.attention == 'on':
        att_model = load_multi_model(models, build_attention)

    # read input sequences
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    seq_id = [] # peptide IDs
    peptide = [] # peptide sequences
    for seq_record in SeqIO.parse(args.seqs, 'fasta'):
        if len(seq_record.seq) > 200 or len(seq_record.seq) < 2 or set(seq_record.seq)-set(aa) != set():
            raise Exception('Invalid Sequences!')
        else:
            seq_id.append(str(seq_record.id))
            peptide.append(str(seq_record.seq))
    
    # generate one-hot encoding input and pad sequences into MAX_LEN long
    X_seq = one_hot_padding(peptide, MAX_LEN)
   
    # ensemble results for the 5 models
    print('\nPredicting...')
    y_score, y_indv_list = ensemble(out_model, X_seq)
    y_class = proba_to_class_name(y_score)

    # get attention scores for each sequence
    if args.attention == 'on':
        attention = get_attention_scores(y_indv_list, att_model, peptide, X_seq)

    # output the predictions
    out_txt = ''
    for i in range(len(seq_id)):
        temp_txt = 'Sequence ID: '+seq_id[i]+'\n'+'Sequence: '+peptide[i]+'\n' \
        +'Score: '+str(y_score[i])+'\n'+'Prediction: '+y_class[i]+'\n'
        if args.attention == 'on':
            temp_txt = temp_txt+'Attention: '+str(list(attention[i]))+'\n'
        temp_txt = temp_txt+'\n'
        print(temp_txt)
        out_txt = out_txt + temp_txt
    
    # save to txt or xlsx    
    if args.out_format is not None:
        print('\nSaving results...')
        out_name = 'AMPlify_results_' + time.strftime('%Y%m%d%H%M%S', time.localtime())
        if (args.out_format).lower() == 'txt':
            out_name = out_name + '.txt'
            if os.path.isfile(args.out_dir + '/' + out_name):
                print('\nUnable to save! File already existed!')
            else:
                out = open(args.out_dir + '/' + out_name, 'w')
                out.write(out_txt)
                out.close()
                print('\nResults saved as: ' + args.out_dir + '/' + out_name)
        else:
            out_name = out_name + '.tsv'
            if os.path.isfile(args.out_dir + '/' + out_name):
                print('\nUnable to save! File already existed!')
            else:
                if args.attention == 'on':
                    out = pd.DataFrame({'Sequence_ID':seq_id,
                                        'Sequence': peptide,
                                        'Score': y_score,
                                        'Prediction': y_class,
                                        'Attention': [list(a) for a in attention]})
                else:
                    out = pd.DataFrame({'Sequence_ID':seq_id,
                                        'Sequence': peptide,
                                        'Score': y_score,
                                        'Prediction': y_class})
                out.to_csv(args.out_dir + '/' + out_name, sep='\t', index=False)
                print('\nResults saved as: ' + args.out_dir + '/' + out_name)
            

if __name__ == "__main__":
    main()