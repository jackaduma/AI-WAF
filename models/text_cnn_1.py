#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-05 14:57:22
LastEditTime: 2022-12-11 13:28:37
LastEditors: Kun
Description: 
FilePath: /AI-WAF/models/text_cnn_1.py
'''

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, AveragePooling1D
from keras import regularizers
from keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

from config import inputLen

def textcnn1(tokenizer, class_num=2):
    kernel_size = [1, 3, 3, 5, 5]
    acti = 'relu'
    my_input = Input(shape=(inputLen,), dtype='int32')
    emb = Embedding(len(tokenizer.word_index) + 1, 20,
                    input_length=inputLen)(my_input)
    emb = SpatialDropout1D(0.2)(emb)

    net = []
    for kernel in kernel_size:
        con = Conv1D(32, kernel, activation=acti, padding="same")(emb)
        con = MaxPool1D(2)(con)
        net.append(con)
    net = concatenate(net, axis=-1)
    # net = concatenate(net)
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(class_num, activation='softmax')(net)
    model = Model(inputs=my_input, outputs=net)
    return model


