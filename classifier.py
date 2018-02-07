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

from models import simple_RNN_model, embed_model
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

y_train = train_data.iloc[:, 2:]
y_train = np.array(y_train)


words_counter = collections.Counter([word for sentence in processed_train_data for word in sentence.split()]+
                                    [word for sentence in processed_test_data for word in sentence.split()])

vocab_size = len(words_counter)
sentence_length = 100

train_text_tokenized, train_text_tokenizer = preprocess.tokenize(processed_train_data, len(words_counter))
test_text_tokenized, test_text_tokenizer = preprocess.tokenize(processed_test_data, len(words_counter))

train_text_tokenized = preprocess.pad(train_text_tokenized, sentence_length)
test_text_tokenized = preprocess.pad(test_text_tokenized, sentence_length)


train_text_tokenized = train_text_tokenized.reshape((train_text_tokenized.shape[-2], -1))
test_text_tokenized = test_text_tokenized.reshape((test_text_tokenized.shape[-2], -1))

print (train_text_tokenized.shape, train_text_tokenized.shape[1:])

# multi-classification RNN model

#model = simple_RNN_model(sentence_length, y_train.shape[1])
model = embed_model(vocab_size, sentence_length, y_train.shape[1])

print (model.summary())

checkpointer = ModelCheckpoint(filepath='embed_model_1.weights.best.hdf5', verbose=1, save_best_only=True)

#train_text_tokenized = train_text_tokenized.reshape((1, -1, train_text_tokenized.shape[-2]))


hist = model.fit(np.array(train_text_tokenized), np.array(y_train), batch_size = 1024, epochs = 100,
                 validation_split = 0.2, callbacks=[checkpointer],
                 verbose=2, shuffle=True)




model.load_weights('embed_model_1.weights.best.hdf5')

predictions = model.predict(test_text_tokenized) # predictions in "is_iceberg"

toxic_label_predictions = predictions[:, 0]
severe_toxic_label_predictions = predictions[:, 1]
obscene_label_predictions = predictions[:, 2]
threat_label_predictions = predictions[:, 3]
insult_label_predictions = predictions[:, 4]
identity_hate_label_predictions = predictions[:, 5]



print (len(predictions), len(processed_test_data))

with open('submission_toxic_2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    writer.writerows(zip(test_data_id, toxic_label_predictions,
                         severe_toxic_label_predictions,
                         obscene_label_predictions,
                         threat_label_predictions,
                         insult_label_predictions,
                         identity_hate_label_predictions))

