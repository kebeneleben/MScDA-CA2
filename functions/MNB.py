import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

def get_training_data(df):
    X = df["cleaned_text"]
    y = df["sentiment_label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_model(df):
    X_train, X_test, y_train, y_test = get_training_data(df)

    vectorizer = CountVectorizer()
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)
    
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(X_train_features, y_train)
    y_pred = naive_bayes_model.predict(X_test_features)
    
    evaluate_model(y_pred, y_test)
    
    return naive_bayes_model, X_train_features, X_test_features, y_train, y_test
    
def evaluate_model(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print()
    
def hypertune_model(model, X_train_features, y_train, X_test_features, y_test):
    param_grid = {'alpha': [0.1, 0.5, 1.0]}
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train_features, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_features)
    
    print("Best parameters:", grid_search.best_params_)
    evaluate_model(y_pred, y_test)
    
    return best_model

def cross_validate(df, model):
    # Define the number of folds for cross-validation
    n_splits = 5
    X = df['cleaned_text']
    y = df['sentiment_label']
    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # Perform cross-validation
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        vectorizer = CountVectorizer()
        X_train_features = vectorizer.fit_transform(X_train)
        X_test_features = vectorizer.transform(X_test)
        
        # Train and evaluate the model on the current fold
        model.fit(X_train_features, y_train)
        y_pred = model.predict(X_test_features)
        
        evaluate_model(y_pred, y_test)