#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-12-11 11:41:58
LastEditTime: 2022-12-11 16:11:24
LastEditors: Kun
Description: 
FilePath: /AI-WAF/models/text_cnn_2.py
'''

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, AveragePooling1D
from keras import regularizers
from keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

from config import inputLen


def textcnn2(tokenizer, class_num=2):
    kernel_size = [3, 5, 7, 9, 11]
    acti = 'relu'
    my_input = Input(shape=(inputLen,), dtype='int32')
    emb = Embedding(len(tokenizer.word_index) + 1, 128,
                    input_length=inputLen)(my_input)
    emb = SpatialDropout1D(0.2)(emb)

    net = []
    for kernel in kernel_size:
        con = Conv1D(8, kernel, activation=acti, padding="valid",
                     kernel_regularizer=l2(0.0005))(emb)
        # con1 = MaxPool1D(2)(con)
        con1 = GlobalAveragePooling1D()(con)
        con2 = GlobalMaxPooling1D()(con)
        net.append(con1)
        net.append(con2)
    net = concatenate(net, axis=-1)
    # net = concatenate(net)
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(256, activation='relu', kernel_regularizer=l2(l=0.001))(net)
    net = Dropout(0.5)(net)
    net = Dense(class_num, activation='softmax',
                kernel_regularizer=l2(l=0.001))(net)
    model = Model(inputs=my_input, outputs=net)
    return model
