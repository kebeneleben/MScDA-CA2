import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def train_model(df):
    X = df['title']
    y = df['sentiment_label']
    
    label_encoder = LabelEncoder()
    #Encode the sentiment labels to 0 and n-1 classes
    y = label_encoder.fit_transform(y)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(seq) for seq in X_train_seq)
    
    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=X_train_seq, vector_size=100, window=5, min_count=1, workers=4)

    # Convert text sequences to word vectors
    X_train_vec = convert_text_to_vec(word2vec_model, X_train_seq)
    X_test_vec = convert_text_to_vec(word2vec_model, X_test_seq)
    # Convert the word vectors to numpy arrays
    X_train_vec = np.array(X_train_vec)
    X_test_vec = np.array(X_test_vec)
    # Pad the training and test data
    X_train_vec = tf.stack(pad_sequences(X_train_vec, maxlen=max_length, padding='post'))
    X_test_vec = tf.stack(pad_sequences(X_test_vec, maxlen=max_length, padding='post'))

    model = Sequential()
    model.add(LSTM(units=128, input_shape=(max_length, 100)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train_vec, tf.stack(y_train), epochs=5, batch_size=32)
    
    y_pred = model.predict_classes()
    
    print(y_pred)
    
    return model, X_train_vec, X_test_vec, y_train, y_test, y_pred

def convert_text_to_vec(word2vec_model, X_train_seq):
    X_vec = []
    
    for seq in X_train_seq:
        seq_vec = []
        for word in seq:
            if word in word2vec_model.wv.key_to_index:
                seq_vec.append(word2vec_model.wv[word])
            else:
                # Replace with the appropriate dimension of word vectors
                seq_vec.append(np.zeros(100))
    X_vec.append(seq_vec)
    return X_vec
    