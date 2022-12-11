#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-05 15:06:08
LastEditTime: 2022-05-05 15:06:30
LastEditors: Kun
Description: 
FilePath: /my_open_projects/AI-WAF/evaluate.py
'''


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_result(y_true, y_pre):
    accuracy = accuracy_score(y_true, y_pre)
    precision = precision_score(y_true, y_pre)
    recall = recall_score(y_true, y_pre)
    f1 = f1_score(y_true, y_pre)
    auc = roc_auc_score(y_true, y_pre)

    print("Accuracy Score is: ", accuracy)
    print("Precision Score is :", precision)
    print("Recall Score is :", recall)
    print("F1 Score: ", f1)
    print("AUC Score: ", auc)
