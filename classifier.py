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
import collections
import csv

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint


training_data_path = "C:\\Users\\Gary\\toxic_comment_classification\\data\\train\\train.csv" 
test_data_path = "C:\\Users\\Gary\\toxic_comment_classification\\data\\test\\test.csv"

# 1. process data

train_data = preprocess.process_data(training_data_path)
test_data = preprocess.process_data(test_data_path)

# 1.1 define cache path

cache_dir = os.path.join("cache", "toxic_comment")
os.makedirs(cache_dir, exist_ok=True)

# 2. process sentences

processed_train_data, processed_test_data = preprocess.preprocess_data(train_data, test_data, cache_dir)

# extract ids

test_data_id = list(test_data['id'])

# extract labels

toxic_label = list(train_data['toxic'])
severe_toxic_label = list(train_data['severe_toxic'])
obscene_label = list(train_data['obscene'])
threat_label = list(train_data['threat'])
insult_label = list(train_data['insult'])
identity_hate_label = list(train_data['identity_hate'])



words_counter = collections.Counter([word for sentence in processed_train_data for word in sentence.split()]+
                                    [word for sentence in processed_test_data for word in sentence.split()])

print (len(processed_train_data))
print (len(words_counter))

train_text_tokenized, train_text_tokenizer = preprocess.tokenize(processed_train_data, len(words_counter))
test_text_tokenized, test_text_tokenizer = preprocess.tokenize(processed_test_data, len(words_counter))

train_text_tokenized = preprocess.pad(train_text_tokenized)
test_text_tokenized = preprocess.pad(test_text_tokenized)


print (train_text_tokenized[1:5])
print (train_text_tokenized.shape)

train_text_tokenized = train_text_tokenized.reshape((train_text_tokenized.shape[-2], -1))

# print an example of one-hot encoded comment


print (train_text_tokenized[1:5])
print (train_text_tokenized.shape)

# multi-classification RNN model


model = Sequential()

model.add(Embedding(286228, 128, input_length=100))
model.add(LSTM(128, dropout=0.2, input_shape=train_text_tokenized.shape[1:], activation="relu"))
model.add(Dense(1, activation='sigmoid'))

print (model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics =['accuracy'])

checkpointer = ModelCheckpoint(filepath='simple_LSTM_1.weights.best.hdf5', verbose=1, save_best_only=True)


# reshape input

#train_text_tokenized = train_text_tokenized.reshape((1, -1, train_text_tokenized.shape[-2]))

hist = model.fit(np.array(train_text_tokenized), np.array(toxic_label), batch_size = 1024, epochs = 100,
                 validation_split = 0.2, callbacks=[checkpointer],
                 verbose=2, shuffle=True)

predictions = [model.predict(np.expand_dims(sentence,axis=0))[0][0] for sentence in test_text_tokenized] # predictions in "is_iceberg"

print (predictions)

print (len(predictions), len(processed_test_data))

with open('submission_toxic_1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "toxic"])
    writer.writerows(zip(test_data_id, predictions))


