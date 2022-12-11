#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-12-11 12:39:25
LastEditTime: 2022-12-11 13:43:06
LastEditors: Kun
Description: 
FilePath: /AI-WAF/models/textcnn_tokenizer.py
'''


import pickle
from keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer

from config import inputLen

########################################################################################

def train_tokenizer(train_datas, test_datas, tokenizer_file_path):
    tokenizer = Tokenizer(num_words=None,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    tokenizer.fit_on_texts(train_datas)
    tokenizer.fit_on_texts(test_datas)
    # print(tokenizer.word_index)
    # # vocal = tokenizer.word_index
    train_datas = tokenizer.texts_to_sequences(train_datas)
    test_datas = tokenizer.texts_to_sequences(test_datas)
    train_datas = pad_sequences(
        train_datas, inputLen, padding='post', truncating='post')
    test_datas = pad_sequences(
        test_datas, inputLen, padding='post', truncating='post')

    with open(tokenizer_file_path, "wb") as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    return tokenizer, train_datas, test_datas


########################################################################################

def train_tokenizer_with_val(train_datas, val_datas, test_datas, tokenizer_file_path):
    tokenizer = Tokenizer(num_words=None,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n',
                          lower=True,
                          split=" ",
                          char_level=False)
    tokenizer.fit_on_texts(train_datas)
    tokenizer.fit_on_texts(val_datas)
    tokenizer.fit_on_texts(test_datas)
    # print(tokenizer.word_index)
    # # vocal = tokenizer.word_index
    train_datas = tokenizer.texts_to_sequences(train_datas)
    val_datas = tokenizer.texts_to_sequences(val_datas)
    test_datas = tokenizer.texts_to_sequences(test_datas)
    train_datas = pad_sequences(
        train_datas, inputLen, padding='post', truncating='post')
    val_datas = pad_sequences(
        val_datas, inputLen, padding='post', truncating='post')
    test_datas = pad_sequences(
        test_datas, inputLen, padding='post', truncating='post')

    with open(tokenizer_file_path, "wb") as tokenizer_file:
        pickle.dump(tokenizer, tokenizer_file)

    return tokenizer, train_datas, val_datas, test_datas
