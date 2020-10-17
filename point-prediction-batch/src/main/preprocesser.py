import datetime

from pandas.core.frame import DataFrame

from db_connector import DBConnector
from details_schema import details_schema


class Preprocesser(object):
    """DBへ投入する前にスクレイピングした作品の詳細情報の前処理を行う"""
    
    @classmethod
    def preprocess_details(cls, details_df: DataFrame) -> DataFrame:
        """前処理(不必要なカラムの削除、カラムの追加、型変換)を実施"""
        details_df = details_df.drop(['allcount', 'gensaku'], axis=1, errors='ignore')
        details_df = details_df.dropna(how='all')
        details_df['bert_train'] =  [0 if idx % 5 == 0 else 1 for idx in details_df.index]
        details_df['ml_train'] =  [1 if idx % 5 == 0 else 0 for idx in details_df.index]
        details_df['predict_point'] = None
        details_df['added_to_es'] = False
        
        # dateをUNIX時刻へ変換する
        for column_name in details_df.columns:
            if column_name in['general_firstup', 'general_lastup', 'novelupdated_at', 'updated_at']:
                details_df[column_name] = details_df[column_name].map(str).map(cls.__date_to_timestamp)
        
        # DataFrameの型変換の実行
        for column_name, column_type in details_schema.items():
            details_df[column_name] = details_df[column_name].astype(column_type)
        
        return details_df
    
    @classmethod
    def __date_to_timestamp(cls, date: str) -> int:
        return int(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp())
