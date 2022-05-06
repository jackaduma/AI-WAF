#!python
# -*- coding: utf-8 -*-
# @author: Kun





import pickle
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, merge, Input, Lambda, Reshape, LSTM, RNN, \
    SimpleRNNCell, SpatialDropout1D, Add, Maximum
from keras.layers import Conv1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, AveragePooling1D
from keras import optimizers
from keras import regularizers
from keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import time

from keras import backend as K

from config import text_cnn_tokenizer_file_path

inputLen = 1024  # 256  # 512


def train_tokenizer(train_datas, test_datas):

    tokenizer = Tokenizer(num_words=None,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    # 通过训练和测试数据集丰富取词器的字典，方便后续操作
    tokenizer.fit_on_texts(train_datas)
    tokenizer.fit_on_texts(test_datas)
    # print(tokenizer.word_index)
    # #获取目前提取词的字典信息
    # # vocal = tokenizer.word_index
    train_datas = tokenizer.texts_to_sequences(train_datas)
    # 通过字典信息将字符转换为对应的数字
    test_datas = tokenizer.texts_to_sequences(test_datas)
    # print(test_apis)
    # 序列化原数组为没有逗号的数组，默认在前面填充,默认截断前面的
    train_datas = pad_sequences(
        train_datas, inputLen, padding='post', truncating='post')
    test_datas = pad_sequences(
        test_datas, inputLen, padding='post', truncating='post')

    with open(text_cnn_tokenizer_file_path, "wb") as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    return tokenizer, train_datas, test_datas


def train_tokenizer_with_val(train_datas, val_datas, test_datas):

    tokenizer = Tokenizer(num_words=None,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    # 通过训练和测试数据集丰富取词器的字典，方便后续操作
    tokenizer.fit_on_texts(train_datas)
    tokenizer.fit_on_texts(val_datas)
    tokenizer.fit_on_texts(test_datas)
    # print(tokenizer.word_index)
    # #获取目前提取词的字典信息
    # # vocal = tokenizer.word_index
    train_datas = tokenizer.texts_to_sequences(train_datas)
    val_datas = tokenizer.texts_to_sequences(val_datas)
    # 通过字典信息将字符转换为对应的数字
    test_datas = tokenizer.texts_to_sequences(test_datas)
    # print(test_apis)
    # 序列化原数组为没有逗号的数组，默认在前面填充,默认截断前面的
    train_datas = pad_sequences(
        train_datas, inputLen, padding='post', truncating='post')
    val_datas = pad_sequences(
        val_datas, inputLen, padding='post', truncating='post')
    test_datas = pad_sequences(
        test_datas, inputLen, padding='post', truncating='post')

    with open(text_cnn_tokenizer_file_path, "wb") as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    return tokenizer, train_datas, val_datas, test_datas


def textcnn1(tokenizer, class_num=2):
    kernel_size = [1, 3, 3, 5, 5]
    acti = 'relu'
    # 可看做一个文件的api集为一句话，然后话中的词总量是6000
    my_input = Input(shape=(inputLen,), dtype='int32')
    emb = Embedding(len(tokenizer.word_index) + 1, 20,
                    input_length=inputLen)(my_input)
    emb = SpatialDropout1D(0.2)(emb)

    net = []
    for kernel in kernel_size:
        # 32个卷积核
        con = Conv1D(32, kernel, activation=acti, padding="same")(emb)
        # 滑动窗口大小是2,默认输出最后一维是通道数
        con = MaxPool1D(2)(con)
        net.append(con)
    # print(net)
    # input()
    net = concatenate(net, axis=-1)
    # net = concatenate(net)
    # print(net)
    # input()
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(256, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(class_num, activation='softmax')(net)
    model = Model(inputs=my_input, outputs=net)
    return model


def textcnn2(tokenizer, class_num=2):
    kernel_size = [1, 3, 5, 7, 9, 11, 13]
    acti = 'relu'
    # 可看做一个文件的api集为一句话，然后话中的词总量是6000
    my_input = Input(shape=(inputLen,), dtype='int32')
    emb = Embedding(len(tokenizer.word_index) + 1, 128,
                    input_length=inputLen)(my_input)
    emb = SpatialDropout1D(0.2)(emb)

    net = []
    for kernel in kernel_size:
        # 128个卷积核
        con = Conv1D(8, kernel, activation=acti, padding="valid",
                     kernel_regularizer=l2(0.0005))(emb)
        # 默认输出最后一维是通道数
        # con1 = MaxPool1D(2)(con)
        con1 = GlobalAveragePooling1D()(con)
        con2 = GlobalMaxPooling1D()(con)
        net.append(con1)
        net.append(con2)
    # print(net)
    # input()
    net = concatenate(net, axis=-1)
    # net = concatenate(net)
    # print(net)
    # input()
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(256, activation='relu', kernel_regularizer=l2(l=0.001))(net)
    net = Dropout(0.5)(net)
    net = Dense(class_num, activation='softmax',
                kernel_regularizer=l2(l=0.001))(net)
    model = Model(inputs=my_input, outputs=net)
    return model
