import sys
import pandas as pd
import numpy as np
import re
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load_data will take messages_filepath ( path to .csv) and categories_filepath (path to .csv), import them as 
    Pandas Dataframes, merge them on the 'id' column, and then return the resulting dataframe.
    
    inputs: 
    messages_filepath (str) to .csv
    categories_filepath (str) to .csv
    
    Outputs:
    pandas DataFrame of merged imported .csv
    """
    # load messages dataset
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    # Note: 'original' column has a lot of nulls, but it looks like the 'message' column still contains data
    # merge datasets
    df = pd.merge(df_messages, df_categories, on='id')
    
    return df

def clean_data(df):
    """
    clean_data will take a df of combined messages+categories data and expand the categories column to a multi-column format
    that can be used for further data pipeline processing and modeling.
    Duplicate rows of information will also be removed from the df and then the resulting output is a cleaned df.
    
    inputs: 
    Pandas DataFrame of merged messages+categories data
    
    Outputs:
    Pandas DataFrame cleaned as described above
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0,:]
    
    # extract a list of new column names for categories.
    category_colnames = row.str.partition('-')[0].tolist()
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.partition('-')[2]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop(columns='categories')

    # concatenate the original dataframe with the new `categories` dataframe - important to specify the axis = 1 to merge column wise!
    df = pd.concat([df, categories], axis=1)

    # remove duplicate rows from df, printing the shape before and after
    print('Initial df shape = {}'.format(df.shape))
    df['is_duplicated'] = df.duplicated()
    df = df[~df['is_duplicated']]
    print('After dupe row removal = {}'.format(df.shape))
    
    return df
    
def save_data(df, database_filename):
    '''
    save_data will take a cleaned Pandas DataFrame and save it as a table in an .sql server.
    
    inputs: Pandas Dataframe, path to database file (str)
    outputs: .db file containing a table with the df information stored in it.
    '''
#     engine = create_engine('sqlite:///{}'.format(database_filename))
# #     df.to_sql('{}_tbl'.format(database_filename[:4]), engine, index=False) 
#     df.to_sql('disaster_table', engine, index=False) 

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('{}'.format("DisasterResponse"), engine, index=False, if_exists='replace')


def main():
    ''' 
    main runs the main functions of process_data.py, consisting of loading + cleaning + saving data to a sql database file
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()