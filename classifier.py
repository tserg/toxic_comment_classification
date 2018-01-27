# -*- coding: utf-8 -*-
"""
Created on Tues Jan 23 15:45:15 2018

Toxic Comments Classifier

@author: Gary
"""

import preprocess
import numpy as np
import pandas
import os
import pickle

training_data_path = "C:\\Users\\Gary\\toxic_comment_classification\\data\\train\\train.csv" 
test_data_path = "C:\\Users\\Gary\\toxic_comment_classification\\data\\test\\test.csv"

train_data = preprocess.process_data(training_data_path)
test_data = preprocess.process_data(test_data_path)

#train_data['comment_text'] = train_data['comment_text'].map(preprocess.process_text)
#test_data['comment_text'] = test_data['comment_text'].map(preprocess.process_text)

cache_dir = os.path.join("cache", "toxic_comment")
os.makedirs(cache_dir, exist_ok=True)


words_train, words_test = preprocess.preprocess_data(train_data, test_data, cache_dir)

toxic_label = list(train_data['toxic'])


print (words_train)