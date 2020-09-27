import sys, os
from datetime import datetime

import pandas as pd

from db_connector import DBConnector


DETAIL_TEXT_DATAFRAME_PATH = './csv/detail_text.csv'
DB_PATH = './sqlite/test.sqlite3'
EXTENSION_PATH = './extensions/libsqlitefunctions.so'


def date_to_timestamp(date):
    return int(datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp())


def df_preprocessing(df):
    df = df.drop(['allcount', 'gensaku'], axis=1, errors='ignore')
    df = df.dropna(how='all')

    for column in df.columns:
        if column in ['title', 'ncode', 'userid', 'writer', 'story', 'keyword']:
            df[column] = df[column].astype(str)
        elif column in['general_firstup', 'general_lastup', 'novelupdated_at', 'updated_at']:
            df[column] = df[column].map(str).map(date_to_timestamp)
        elif column in ['text', 'predict_point']:
            pass
        else:
            df[column] = df[column].astype(int)
    
    
    df['text'] = df['text'].astype(str) if 'text' in df.columns else 'Nan'
    df['predict_point'] = df['predict_point'].astype(str) if 'predict_point' in df.columns else 'Nan'
    
    return df


def main():
    db_connector = DBConnector(DB_PATH, EXTENSION_PATH)
    detail_text_df = pd.read_csv(DETAIL_TEXT_DATAFRAME_PATH)
    detail_text_df = df_preprocessing(detail_text_df)
    db_connector.initialize_details_by_dataframe(detail_text_df)
    db_connector.close()



if __name__ == '__main__':
    main()