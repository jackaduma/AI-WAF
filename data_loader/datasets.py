#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-05 15:57:10
LastEditTime: 2022-12-11 14:05:40
LastEditors: Kun
Description: 
FilePath: /AI-WAF/data_loader/datasets.py
'''

import os
import codecs
import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import DATA_DIR

def load_data(validate=False):
    good_text_list = []
    with codecs.open(os.path.join(DATA_DIR, "goodqueries.txt"), mode="r", encoding="utf-8") as good_file:
        for line in good_file.readlines():
            text = line.strip()
            good_text_list.append(text)
    good_df = pd.DataFrame()
    good_df['text'] = good_text_list               
    good_df["label"] = 0
    print("good_df: ", good_df.shape)
    
    #! trick for small test
    good_df = good_df.sample(50000)

    bad_text_list = []
    with codecs.open(os.path.join(DATA_DIR, "badqueries.txt"), mode="r", encoding="utf-8") as bad_file:
        for line in bad_file.readlines():
            text = line.strip()
            bad_text_list.append(text)

    bad_df = pd.DataFrame()
    bad_df['text'] = bad_text_list
    bad_df["label"] = 1
    print("bad_df: ", bad_df.shape)

    df = pd.concat([good_df, bad_df])
    print("total: ", df.shape)

    df = df[(df['label'] == 0) | (df['label'] == 1)]
    print("filter label: 1 or 0 ", df.shape)

    df = df.drop_duplicates(subset='text')
    print("after drop_duplicates df: ", df.shape)

    df["label"] = pd.to_numeric(df["label"])

    df = df.sample(frac=1).reset_index(drop=True)
    df.rename(columns={'text': 'X', 'label': 'y'}, inplace=True)

    datas = df['X'].tolist()
    labels = df['y'].tolist()

    if validate is True:
        train_datas, test_datas, train_labels, test_labels = train_test_split(
            datas, labels, test_size=0.2)

        val_datas, test_datas, val_labels, test_labels = train_test_split(
            test_datas, test_labels, test_size=0.5)

        train_labels = np.eye(2)[train_labels]
        val_labels = np.eye(2)[val_labels]
        test_labels = np.eye(2)[test_labels]

        return train_datas, val_datas, test_datas, train_labels, val_labels, test_labels

    else:
        train_datas, test_datas, train_labels, test_labels = train_test_split(
            datas, labels, test_size=0.2)

        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)

        return train_datas, test_datas, train_labels, test_labels

