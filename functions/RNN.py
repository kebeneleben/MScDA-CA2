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
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from tensorflow.keras import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


corpus = api.load('text8')
vocab_size = 100
max_length = None
vector_size = 300
tokenizer = Tokenizer()
label_encoder = LabelEncoder()
word2vec_model = Word2Vec(sentences=corpus, vector_size=vector_size, window=5, min_count=1, workers=4)

def preprocess_split(X_train, X_test, y_train, y_test):
    '''Processes and splits the data in to training sets and test sets. Uses word2vec to create vectorized texts'''
    global max_length
    global vocab_size
    global vector_size
    #Encode the sentiment labels to 0 and n-1 classes
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(seq) for seq in X_train_seq)
    
    # Train Word2Vec model
#     word2vec_model = api.load("word2vec-google-news-300")# Word2Vec(sentences=corpus, vector_size=vector_size, window=5, min_count=1, workers=4)
    vector_size = word2vec_model.vector_size
    
    # Convert text sequences to word vectors
    X_train_vec = convert_text_to_vec(X_train_seq, max_length)
    X_test_vec = convert_text_to_vec(X_test_seq, max_length)
    
    return X_train_vec, X_test_vec, y_train, y_test

def train_model(df):
    '''Creates and trains an RNN model and shows the performance. It also returns the split data as well as the model to be used in hyperparameter tuning and cross validation.'''
    global max_length
    global vocab_size
    global vector_size
    
    X = df['cleaned_text']
    y = df['sentiment_label']
    y = label_encoder.fit_transform(y)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_vec, X_test_vec, y_train, y_test = preprocess_split(X_train, X_test, y_train, y_test)
    rnn_model = KerasClassifier(build_fn=create_rnn_model, epochs=10, batch_size=32, verbose=0)

    # Fit the model
    rnn_model.fit(X_train_vec, y_train)

    # Evaluate the model
    y_pred = rnn_model.predict(X_test_vec)
    
    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    # Compute precision
    precision = precision_score(y_test, y_pred, average="macro")
    # Compute recall
    recall = recall_score(y_test, y_pred, average="macro")
    # Compute F1 score
    f1 = f1_score(y_test, y_pred, average="macro")
    
    print("Accuracy: ", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    
    return rnn_model, X_train_vec, X_test_vec, y_train, y_test

def create_rnn_model():
    '''Creates the RNN model'''
    model = Sequential()
    model.add(LSTM(units=128, input_shape=(max_length, vector_size)))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', metrics.Precision(), metrics.Recall()])
    return model

def convert_text_to_vec(X_train_seq, max_length):
    '''Converts the text sequnce into a vector'''
    X_vec = []
    # Loop through the list of sequence
    for seq in X_train_seq:
        seq_vec = []
        # Loop through the text index in a sequence
        for word_id in seq:
            try:
                # If the word exists, get the vectors (relationship to other words) or the word
                word = tokenizer.index_word[word_id]
                word_vector = word2vec_model.wv[word]
#                 print(word, word_vector)
            except KeyError:
                # Just set it to zero, meaning the word doesn't exist
                word_vector = np.zeros(vector_size)
            seq_vec.append(word_vector)
        
        X_vec.append(seq_vec)
        
    X_vec = pad_sequences(X_vec, maxlen=max_length, padding="post", dtype='float32')
    X_vec = X_vec.reshape(X_vec.shape[0], X_vec.shape[1], -1)
    return X_vec

def hypertune(model, X_train_vec, y_train, X_test_vec, y_test):
    '''Train the model on the parameters given'''
    parameters = {'epochs': [10, 20, 30], 'batch_size': [32, 64, 128]}
    grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=5)
    grid_search.fit(X_train_vec, y_train)
    return grid_search

def cross_validate(df, model):
    '''Perform cross validation using StratifiedKFold on the best model'''
    # Define the number of folds for cross-validation
    n_splits = 5
    X = df['cleaned_text']
    y = df['sentiment_label']
    y = label_encoder.fit_transform(y)
    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # Perform cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train_vec, X_test_vec, y_train, y_test = preprocess_split(X_train, X_test, y_train, y_test)
        
        # Train and evaluate the model on the current fold
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Compute precision
        precision = precision_score(y_test, y_pred, average="macro")
        # Compute recall
        recall = recall_score(y_test, y_pred, average="macro")
        # Compute F1 score
        f1 = f1_score(y_test, y_pred, average="macro")
        
        print("Accuracy: ", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)