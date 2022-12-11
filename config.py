#!python
# -*- coding: utf-8 -*-
# @author: Kun

'''
Author: Kun
Date: 2022-05-05 14:59:20
LastEditTime: 2022-12-11 13:52:25
LastEditors: Kun
Description: 
FilePath: /AI-WAF/config.py
'''


import os

DATA_DIR = "./data"
CACHE_DIR = "./cache"

########################################################################################

class_num = 2
inputLen = 1024  # 256  # 512

########################################################################################

textcnn1_model_dir = os.path.join(CACHE_DIR, "textcnn1")
os.makedirs(textcnn1_model_dir, exist_ok=True)

textcnn2_model_dir = os.path.join(CACHE_DIR, "textcnn2")
os.makedirs(textcnn2_model_dir, exist_ok=True)

textcnn3_model_dir = os.path.join(CACHE_DIR, "textcnn3")
os.makedirs(textcnn3_model_dir, exist_ok=True)

########################################################################################