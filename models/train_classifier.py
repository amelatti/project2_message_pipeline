import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle

import sqlite3
from sqlalchemy import create_engine

def load_data(database_filepath):
    '''
    load_data will import a sql .db file of data to a Pandas DataFrame with a table_name of 'DisasterResponse'
    Input: database_filepath (str) - the path of the database
    Outputs: 
    X (list) - list of the disaster messages that will be analyzed by the machine learning algorithm
    Y (df) - Pandas DataFrame of each of the columns for classification values for the disaster messages in X
    category_names (list) - list of the 36 different category names for the classification df

    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name='DisasterResponse', con=engine)

    # Pull the X data (messages) and Y data (classification results) for use later on in training/testing.
    X = df['message'].tolist()
    Y = df[df.columns[4:]].astype(int)
    category_names = Y.columns.tolist()
    return [X, Y, category_names]

def tokenize(text):

    '''
    tokenize will take a str input and clean and convert it into individual tokens of text to be used in the NLP pipeline.
    Input: text (str)
    Outputs: 
    tokens - array of strings that have been cleaned and converted.

    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    
    # Remove any punctuation and make the strings lowercase
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize the individual words in each string
    tokens = word_tokenize(text)
    
    # Lemmatize (reduce down to root word) for each word in tokens but only if the word is not in stop_words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

def build_model():

    '''
    build_model will make the MLP consisting of a countvectorizer (using our tokenize function as a tokenizer), a TfidfTransofmer,
    and a multioutputclassifier using randomforest estimator.
    We then create a gridsearch object to attempt to determine the optimum parameters for the pipeline.
    Output: model to be used to fit and evaluate data on.

    '''
    # building initial pipeline with countvectorizer, and tfidf for transformations, and then classifier is multioutputclassifier with random forest!
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
        
    ])

    # Note: I have removed the below parameters as they were too slow and the model was not training!
    # specify parameters for grid search to try and tweak for better results
    # parameters = {
    #     'clf__estimator__min_samples_split': [2, 3, 4],
    #     'clf__estimator__n_estimators': [25, 50, 100],

    # }
    parameters = {'clf__estimator__n_estimators':[5]}

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model takes the model generated in build_model and applies X and Y testing data to it 
    so that the results of the model's training can be evaluated.
    Inputs: model
    X_test - list of messages to test on
    Y_test - array of classification results to compare predictions against

    '''
    y_pred = model.predict(X_test)

    print("\nBest Parameters Identified:", model.best_params_)
    accuracy = (y_pred == Y_test).mean()
    print('Accuracy identified at: {}'.format(accuracy))
    for n, col in enumerate(Y_test):
        print('The classification report for column "{}" is...'.format(col))
        print(classification_report(Y_test[col], y_pred[:,n]))

def save_model(model, model_filepath):
    '''
    save_model saves the created model in a .pkl file so it can be saved, distributed, and used by other people or web apps.
    Inputs: model to save
    model_filepath (path you want to save the model's .pkl to)

    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    '''
    main handles the flow of the MLP using the above defined functions.
    It imports data from an SQL .db; 
    cleans the data and sets it up for modeling using train_test_split;
    builds a model and then trains the model on the training data;
    evaluates the model;
    saves the model and exports it.

    Inputs: train_classifier.py reference, database_filepath, model_filepath
    outputs: trained and evaluated model in a .pkl file

    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()