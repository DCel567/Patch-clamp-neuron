# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:53:49 2019

@author: ncelik34
"""
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# Importing the Keras libraries and packages
import tensorflow as tf
from tensorflow_addons.metrics import F1Score
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.utils import to_categorical
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import math


batch_size = 256

Dname = 'intervals/0.0001_6.5.csv'

df30 = pd.read_csv(Dname, header=None)
dataset = df30.values
dataset = dataset.astype('float64')
timep = dataset[:, 0]
maxer = np.amax(dataset[:, 2])
print(maxer)
maxeri = maxer.astype('int')
maxchannels = maxeri
idataset = dataset[:, 2]
idataset = idataset.astype(int)

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


def mcor(y_true, y_pred):
    # matthews_correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def smoothing(raw, pred):
    channel_nums = [0] * 6
    amount = len(pred)
    for i in pred:
        channel_nums[i] += 1

    first = 0
    for i in range(len(channel_nums)):
        first = i if channel_nums[i] == max(channel_nums) else first
    print(channel_nums, first)

    # first point:
    if pred[0] < first:
        pred[0] = np.mean(raw)
    else:
        raw[0] = raw[0]/2

    # rest points:
    start = 0
    for i in range(1, amount):
        if pred[i] < first <= pred[i - 1]:
            start = i
        if pred[i] >= first and pred[i-1] < first:
            stop = i
            for j in range(start, stop):
                raw[j] = (raw[start-1] + raw[stop])/2
        if i == amount-1 and pred[i] < first:  #
            for j in range(start, amount):  #
                raw[j] = raw[start]  #
        if pred[i] >= first:
            raw[i] = raw[i]/2

    m = min(raw)
    for i in range(amount):
        raw[i] = raw[i]-m
    return raw


def alter_smoothing(raw, pred):
    channel_nums = [0] * 6
    amount = len(pred)
    for i in pred:
        channel_nums[i] += 1

    first = 0
    for i in range(len(channel_nums)):
        first = i if channel_nums[i] == max(channel_nums) else first
    print(channel_nums, first)

    # first point:
    if pred[0] < first:
        pred[0] = np.mean(raw)

    # rest points:
    start = 0
    div_sum = 0
    div_am = 0
    for i in range(1, amount):
        if pred[i] < first <= pred[i - 1]:
            start = i
        if pred[i] >= first and pred[i-1] < first:
            stop = i
            for j in range(start, stop):
                old_raw = raw[j]
                raw[j] = (raw[start-1] + raw[stop])/2
                div_sum += old_raw - raw[j]
                div_am += 1

    div_sum = div_sum / div_am
    for i in range(amount):
        if pred[i] >= first:
            raw[i] = raw[i] - div_sum

    m = min(raw)
    for i in range(amount):
        raw[i] = raw[i]-m
    return raw


def half_smoothing(raw, pred):
    channel_nums = [0] * 6
    amount = len(pred)
    for i in pred:
        channel_nums[i] += 1

    first = 0
    for i in range(len(channel_nums)):
        first = i if channel_nums[i] == max(channel_nums) else first
    print(channel_nums, first)

    # first point:
    if pred[0] < first:
        pred[0] = np.mean(raw)

    # rest points:
    start = 0
    stop = 0  #
    for i in range(1, amount):
        if pred[i] < first <= pred[i - 1]:
            start = i
        if pred[i] >= first and pred[i-1] < first:
            stop = i
            for j in range(start, stop):
                raw[j] = (raw[start-1] + raw[stop])/2
        if i == amount-1 and pred[i] < first:  #
            for j in range(start, amount):  #
                raw[j] = raw[start]  #

    m = min(raw)
    for i in range(amount):
        raw[i] = raw[i] - m
    return raw


train_size = int(len(dataset))

in_train = dataset[:, 1]
target_train = idataset

in_train = in_train.reshape(len(in_train), 1, 1, 1)

loaded_model = load_model('model/my_own_deepchanel_to5_with534_val.h5', custom_objects={
                          'mcor': mcor, 'precision': precision, 'recall': recall, 'F1Score': f1, 'auc': auc})

loaded_model.summary()

c = np.argmax(loaded_model.predict(in_train, batch_size=batch_size, verbose=True), axis=-1)

raw_data = dataset[:, 1]


raw_data = smoothing(raw_data, c)
in_train = raw_data.reshape(len(raw_data), 1, 1, 1)
c = np.argmax(loaded_model.predict(in_train, batch_size=batch_size, verbose=True), axis=-1)

lenny = 0
ulenny = 6000

plt.figure(figsize=(30, 6))
plt.subplot(2, 1, 1)

plt.plot(dataset[lenny:ulenny, 1], color='blue')
plt.ylabel('Значение тока')
# plt.title("The raw test")

plt.subplot(2, 1, 2)
plt.plot(c[lenny:ulenny], color='red')

plt.xlabel('Временные точки')
plt.ylabel('Открытые каналы')
plt.legend()
plt.show()