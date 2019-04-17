#!/usr/bin/python3
import argparse
import sys
import numpy as np
import tensorflow as tf
import keras.backend as K
import prepare_similarity_matrix

matrix = tf.constant(prepare_similarity_matrix.loadBlosum62Matrix())

def mse_l1(y_true, y_pred):
    absCol = tf.abs(matrix)
    maxByCol = tf.reduce_max(absCol, axis=0, keepdims=True)
    l1Norm = tf.reduce_sum(absCol, axis=0, keepdims=True)
    blosumMatrix = tf.divide(tf.subtract(maxByCol, matrix), l1Norm)

    y_true_argmax = K.argmax(y_true, axis=2)
    y_pred_argmax = K.argmax(y_pred, axis=2)
    result = tf.gather_nd(blosumMatrix, tf.stack((y_true_argmax, y_pred_argmax), -1))
    resultCorrect = K.cast(result, K.dtype(y_true))
    
    # Non-automatic masking applied when necessary.
    constant_zero_masker = K.zeros_like(y_true_argmax)
    masker = tf.math.logical_not(tf.equal(y_true_argmax, constant_zero_masker))
    masker_int = tf.cast(masker, dtype=tf.int32)
    masker_float32 = K.cast(masker_int, dtype=K.dtype(y_true))
    resultFinal = tf.multiply(masker_float32, resultCorrect)

    return K.mean(K.dot(resultFinal, K.square(y_pred-y_true)), axis=-1)
    
def mse_l2(y_true, y_pred):
    blosumMatrix = K.l2_normalize(matrix, axis=-1)

    y_true_argmax = K.argmax(y_true, axis=2)
    y_pred_argmax = K.argmax(y_pred, axis=2)
    result = tf.gather_nd(blosumMatrix, tf.stack((y_true_argmax, y_pred_argmax), -1))
    resultCorrect = K.cast(result, K.dtype(y_true))
    
    # Non-automatic masking applied when necessary.
    constant_zero_masker = K.zeros_like(y_true_argmax)
    masker = tf.math.logical_not(tf.equal(y_true_argmax, constant_zero_masker))
    masker_int = tf.cast(masker, dtype=tf.int32)
    masker_float32 = K.cast(masker_int, dtype=K.dtype(y_true))
    resultFinal = tf.multiply(masker_float32, resultCorrect)

    return -K.mean(K.dot(resultFinal, K.square(y_pred-y_true)), axis=-1) 

def mse_offset(y_true, y_pred):
    blosumMatrix = tf.reduce_max(matrix) - matrix

    y_true_argmax = K.argmax(y_true, axis=2)
    y_pred_argmax = K.argmax(y_pred, axis=2)
    result = tf.gather_nd(blosumMatrix, tf.stack((y_true_argmax, y_pred_argmax), -1))
    resultCorrect = K.cast(result, K.dtype(y_true))
    
    # Non-automatic masking applied when necessary.
    constant_zero_masker = K.zeros_like(y_true_argmax)
    masker = tf.math.logical_not(tf.equal(y_true_argmax, constant_zero_masker))
    masker_int = tf.cast(masker, dtype=tf.int32)
    masker_float32 = K.cast(masker_int, dtype=K.dtype(y_true))
    resultFinal = tf.multiply(masker_float32, resultCorrect)

    return K.mean(K.dot(resultFinal, K.square(y_pred-y_true)), axis=-1)