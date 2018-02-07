from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, GRU, Bidirectional

def simple_RNN_model(input_length, output_shape):
    
    model = Sequential()

    model.add(LSTM(256, dropout=0.2, input_dim=1, input_length=input_length, activation="relu"))
    model.add(Dense(output_shape, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics =['accuracy'])
    
    return model

def embed_model(vocab_size, input_length, output_shape):
    
    model = Sequential()
    
    model.add(Embedding(vocab_size, 256, input_length = input_length))
    
    model.add(LSTM(256, dropout=0.2))
    
    model.add(Dense(output_shape, activation = 'sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def bd_model(input_length, output_shape):
    
    model = Sequential()
    
    model.add(Bidirectional(LSTM(256), input_length=input_length))
    
    model.add(Dense(output_shape, activation='sigmoid'))
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model