import sys
# import libraries
import nltk
import re
nltk.download(['punkt','wordnet','averaged_perceptron_tagger'])

import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report

import pickle

def load_data(database_filepath):
    '''
    input:
        - database_filepath: the database filepath in string format
    output:
        - X: messages inforamtion 
        - Y: categories of each information
        - column_names: the unique category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(database_filepath, engine)
    X = df['message']
    Y = df.iloc[:,3:]
    column_names = Y.columns
    
    return X, Y, column_names

def tokenize(text):
    '''
    Input：
        - text: the message information in string format
    output
        - clean_tokens: the tokens after transfering , tokenizing and lemmmatizing messages
    '''
    url_regex =  'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizar
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
'''
Input: 
    - BaseEstimator: BaseEstimator importing from sklearn.base
    - TransformerMixin: TransformerMixin importing from sklearn.base
'''
    
    def starting_verb(self, text):
        '''
        Input:
            -   text: the text string
        output:
            - Boolen value: If the starting word is a verb, return true.
                            Otherwise, return false
        '''
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB','VBP'] or first_word == 'RT':
                return True
        return False
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, X):
        '''
        Input:
            - X: the dataframe contains the messages in each row
        Output:
            - pd.DataFrame(X_tagged): the dataframe with boolen value indicating whether the starting word is a verb or not 
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    input:
      - None
    output:
      - the ML pipeline after using GridSearch for selecting optimal parameters
    '''
    pipeline = Pipeline([
        ('features',FeatureUnion([
            ('text_pipeline',Pipeline([
                ('vect', CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())

        ])),

        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__max_depth':[10,20,30],'clf__estimator__min_samples_split':[2,5,10]}

    cv_updated = GridSearchCV(pipeline, param_grid = parameters)

    return cv_updated


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    The function will print out classification report with accuracy for each categories
    input:
        - model: the ML pipeline
        - X_test: messages testing dataset
        - Y_test: category testing dataset
        - category_names: all unique category names
    output:
        - None
    '''
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = category_names

    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], y_pred[col]))  


def save_model(model, model_filepath):
    '''
    The funcation will save the model in target path
    Input:
        - model: the pipeline we want to save
        - model_filepath: the path we want to save the pipeline
    Output:
        - None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
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
