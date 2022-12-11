#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-12-11 11:42:07
LastEditTime: 2022-12-11 13:29:17
LastEditors: Kun
Description: 
FilePath: /AI-WAF/models/text_cnn_3.py
'''

from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, AveragePooling1D
from tensorflow.keras.regularizers import l2

from config import inputLen


########################################################################

def textcnn3(tokenizer, class_num=2):
    kernel_size = [2, 3, 4]
    acti = 'relu'
    my_input = Input(shape=(inputLen,), dtype='int32')
    emb = Embedding(len(tokenizer.word_index)+1, 128,
                    input_length=inputLen)(my_input)

    net = []
    for kernel in kernel_size:
        # 32个卷积核
        con = Conv1D(512, kernel, activation=acti, padding="same")(emb)
        net.append(con)
    net = concatenate(net, axis=1)
    # net = concatenate(net)
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(class_num, activation='softmax')(net)
    model = Model(inputs=my_input, outputs=net)
    return model

def textcnn3_with_l2(tokenizer, class_num=2):
    kernel_size = [2, 3, 4]
    acti = 'relu'
    my_input = Input(shape=(inputLen,), dtype='int32')
    emb = Embedding(len(tokenizer.word_index)+1, 128,
                    input_length=inputLen)(my_input)

    net = []
    for kernel in kernel_size:
        con = Conv1D(512, kernel, activation=acti, padding="same", kernel_regularizer=l2(0.0005))(emb)
        net.append(con)
    net = concatenate(net, axis=1)
    # net = concatenate(net)
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(class_num, activation='softmax', kernel_regularizer=l2(l=0.001))(net)
    model = Model(inputs=my_input, outputs=net)
    return model

########################################################################

