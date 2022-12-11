#!python
# -*- coding: utf-8 -*-
# @author: Kun


'''
Author: Kun
Date: 2022-05-05 14:57:30
LastEditTime: 2022-12-11 14:03:32
LastEditors: Kun
Description: 
FilePath: /AI-WAF/train.py
'''

import os
import time
import argparse
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint

from models.textcnn_tokenizer import train_tokenizer, train_tokenizer_with_val
from models.text_cnn_1 import textcnn1
from models.text_cnn_2 import textcnn2
from models.text_cnn_3 import textcnn3
from data_loader.datasets import load_data
from evaluate import evaluate_result

from config import textcnn1_model_dir, textcnn2_model_dir, textcnn3_model_dir, class_num

def train(batch_size, epoch_num, model_option):
    print("Start Train Job! ")
    start = time.time()

    if model_option == 1:
        tokenizer_file_path = os.path.join(textcnn1_model_dir, "tokenizer.pickle")

    elif model_option == 2:
        tokenizer_file_path = os.path.join(textcnn2_model_dir, "tokenizer.pickle")

    elif model_option == 3:
        tokenizer_file_path = os.path.join(textcnn3_model_dir, "tokenizer.pickle")

    else:
        raise Exception("not supported model_option: {}".format(model_option))


    train_datas, val_datas, test_datas, train_labels, val_labels, test_labels = load_data(
        validate=True)
    tokenizer, train_datas, val_datas, test_datas = train_tokenizer_with_val(
        train_datas, val_datas, test_datas, tokenizer_file_path)
        
    if model_option == 1:
        model = textcnn1(tokenizer, class_num)
        model_save_path = os.path.join(textcnn1_model_dir, 'model.h5')

    elif model_option == 2:
        model = textcnn2(tokenizer, class_num)
        model_save_path = os.path.join(textcnn2_model_dir, 'model.h5')

    elif model_option == 3:
        model = textcnn3(tokenizer, class_num)
        model_save_path = os.path.join(textcnn3_model_dir, 'model.h5')
        
    else:
        raise Exception("not supported model_option: {}".format(model_option))

    print(model.summary())

    # optimizer = Adam(learning_rate=1e-3)
    
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(
        model_save_path, save_best_only=True, save_weights_only=True)
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='min', baseline=None,
                              restore_best_weights=True)
    model.fit(train_datas, train_labels, epochs=epoch_num, batch_size=batch_size,
              validation_data=(val_datas, val_labels), callbacks=[checkpoint, earlystop])

    end = time.time()
    print("Over train job in %f s" % (end-start))

    model.load_weights(model_save_path)
    labels_true = test_labels
    print("labels_true: ", labels_true.shape)

    labels_pre = model.predict(test_datas)
    print("labels_pre: ", labels_pre.shape)

    labels_pre = np.array(labels_pre).round()

    def to_y(labels):
        y = []
        for i in range(len(labels)):
            label = labels[i]

            if label[0] == 1:
                y.append(0)

            elif label[1] == 1:
                y.append(1)

            else:
                raise Exception("not supported result: {}".format(label))

        return y
    y_true = to_y(labels_true)
    y_pre = to_y(labels_pre)

    evaluate_result(y_true, y_pre)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="AI_WAF TextCNN training...")
    arg_parser.add_argument("--option", "-o", type=int,
                            required=True, default=2, help="1 or 2 or 3")
    arg_parser.add_argument("--batch_size", "-b", type=int,
                            required=True, default=32, help="batch size 8 16 32 64 ...")
    arg_parser.add_argument("--epoch", "-e", type=int,
                            required=False, default=10, help="epoch num  10 20 ...")

    args = arg_parser.parse_args()
    model_option = args.option
    batch_size = args.batch_size
    epoch_num = args.epoch

    train(batch_size, epoch_num, model_option)