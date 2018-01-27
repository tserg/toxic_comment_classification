# -*- coding: utf-8 -*-
"""
Created on Tues Jan 23 15:45:15 2018

Toxic Comments Classifier

@author: Gary
"""

import pandas
import csv
import re
import nltk
import os
import pickle
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

def process_data(filepath):
    '''
    param:
        @filepath: file path of data
    
    return:
        pandas datafrane
       
    '''
    with open(filepath, 'r', encoding='utf-8') as f:
        data = pandas.read_csv(f)
        
    data['comment_text'] = data['comment_text'].str.lower()
        
    return data

comment_count = 0

def process_text(text):
    
    # remove new line tags
    text=re.sub("\n", " ", text)
    
    # remove punctuation except apostrophe
    text=re.sub(r"[^a-zA-Z0-9']", " ", text)
    
    # tokenise
    text = text.split()
    
    # remove stopwords
    
    words = [word for word in text if word not in stopwords.words("English")]
    
    global comment_count
    comment_count+=1
    if comment_count % 1000 == 0:
        print("Number of comments processed to words = " + str(comment_count))
    
    return words

def preprocess_data(data_train, data_test,
                    cache_dir, cache_file='preprocessed_data.pkl'):
    
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print ("Read preprocessed data from cache file:", cache_file)
        
        except:
            pass 
    
    if cache_data is None:
        
        words_train  = list(map(process_text, data_train['comment_text']))
        words_test = list(map(process_text, data_test['comment_text']))
        
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print ("Wrote preprocessed data to cache file:", cache_file)
        
    else:
            
        words_train, words_test = (cache_data['words_train'], cache_data['words_test'])
    
    return words_train, words_test
'''
def extract_BoW_features(words_train, words_test, vocabulary_size=5000,
                         cache_dir, cache_file="bow_features.pkl"):
    
    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), 'rb') as f:
                cache_data = joblib.load(f)
            print ("Read features from cache file:", cache_file)
        except:
            pass
        
    # If cache_file is missing
    
    if cache_data is None:
        
    
        vectorizer = CountVectorizer(max_features=vocabulary_size, preprocessor=lambda x: x,
                                     tokenizer=lambda x: x)
        
        features_train = vectorizer.fit_transform(words_train).to_array()
        
        features_test = vectorizer.fit_transform(words_test).to_array()
        
        if cache_file is not None:
              
            vocabulary = vectorizer.vocabulary_ 
            cache_data = dict(features_train=features_train, features_test=features_test,
                                  vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), 'wb') as f:
                joblib.dump(cache_data, f)
            print ("Wrote features to cache file:", cache_file)
            
    else: # unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                                                     cache_data['features_test'],
                                                     cache_data['vocabulary'])
        
    return features_train, features_test, vocabulary
'''
