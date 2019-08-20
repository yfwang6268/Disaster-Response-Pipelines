import sys

# import libraries
import pandas as pd
import re


from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    '''
    inputs:
       - messages_filepath: the file path of messages.csv
       - categories_filepath: the file path of categories.csv

    Outputs:
       - df: the dataframe that contains each messages and categories
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath, dtype = str)
    categories = pd.read_csv(categories_filepath,dtype = str)
    df = pd.merge(messages,categories, on = ['id'])
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(pat = ";",expand = True)
    # select the first row of the categories dataframe
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop(['categories'],axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    return df


def clean_data(df):
    '''
    inputs:
       - df: the dataframe contains messages and categories
    Outputs:
       - df: the dataframe after removing duplicated and na value and correcting wrong information
    
    '''
    # drop duplicates
    df.drop(df[df.duplicated() == True].index,axis = 0,inplace = True)
    # drop the redundent column 'original'
    df.drop('original', axis = 1, inplace = True)
    # Given the same number of rows in each column, drop the rows contain na
    df.dropna(axis = 0, inplace = True)
    # replace value 2 to value 1
    df[df['related'] == 2]
    df.replace(2,1,inplace = True)
    return df


def save_data(df, database_filename):
     '''
     This function is to save the data into SQL database
     inputs: 
        -df: the dataframe want to save
        -database_filename: the file name with file path
     outputs: null
     '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, con=engine, index=False)
    pass  


def main():
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
