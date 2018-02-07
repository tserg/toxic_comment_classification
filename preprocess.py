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
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
    
    # remove numbers and punctuation except apostrophe
    text=re.sub(r"[^a-zA-Z]", " ", text)
    
      
    # reduce words to their stems
    # words = [PorterStemmer().stem(w) for w in words]

    global comment_count
    comment_count+=1
    if comment_count % 1000 == 0:
        print("Number of comments processed to words = " + str(comment_count))
    
    return text

def preprocess_data(data_train, data_test,
                    cache_dir, cache_file='preprocessed_data2.pkl'):
    
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
def extract_BoW_features(words_train, words_test, cache_dir, vocabulary_size=10000,
                         cache_file="bow_features.pkl"):
    
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
        
        # removed toarray() from vectorizer.fit_transform because of MemoryError
        
        features_train = vectorizer.fit_transform(words_train) 
        
        features_test = vectorizer.fit_transform(words_test)
        
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

def tokenize(x, vocab_count):
    """
    Tokenize x
    :param x: List of sentences/strings to be tokenized
    :param vocab_count: Number of words in the vocabulary
    :return: Tuple of (tokenized x data, tokenizer used to tokenize x)
    """
    # TODO: Implement
    
    token = Tokenizer(num_words=vocab_count)
    token.fit_on_texts(x)

    words = token.texts_to_sequences(x)
    
    return words, token

def pad(x, length):
    """
    Pad x
    :param x: List of sequences.
    :param length: Length to pad the sequence to.  If None, use length of longest sequence in x.
    :return: Padded numpy array of sequences
    """
    padded_x = pad_sequences(x, maxlen=length, dtype='int32', padding='post', truncating='post', value=0.)
    
    return padded_x
