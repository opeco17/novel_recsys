import json
import sys
from typing import List, Tuple, Any
sys.path.append('..')

from bs4 import BeautifulSoup
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import MeCab
import MySQLdb
import pandas as pd
from pandas.core.frame import DataFrame
import requests

from config import Config


DB_BATCH_SIZE = Config.DB_BATCH_SIZE
DB_HOST_NAME = Config.DB_HOST_NAME
DB_PORT = Config.DB_PORT
ELASTICSEARCH_BATCH_SIZE = Config.ELASTICSEARCH_BATCH_SIZE
ELASTICSEARCH_HOST_NAME = Config.ELASTICSEARCH_HOST_NAME
FEATURE_EXTRACTION_URL = Config.FEATURE_EXTRACTION_URL
H_DIM = Config.H_DIM
POINT_PREDICTION_URL = Config.POINT_PREDICTION_URL


class DBConnector(object):
    """DBとの接続を担うクラス"""

    chunksize = DB_BATCH_SIZE
    host_name = DB_HOST_NAME
    port = DB_PORT

    @classmethod
    def get_conn_and_cursor(cls) -> Tuple[Any, Any]:
        """DBへ接続するメソッド"""
        conn = MySQLdb.connect(
            host = cls.host_name,
            port = cls.port,
            user = 'root',
            password = 'root',
            database = 'maindb',
            use_unicode=True,
            charset="utf8"
        )
        cursor = conn.cursor()
        return conn, cursor

    @classmethod
    def add_details(cls, conn, cursor, details_df):
        """作品の詳細情報全てを追加"""
        cursor.execute("SHOW columns FROM details")
        columns_of_details = [column[0] for column in cursor.fetchall()]
        useable_details_df = details_df[columns_of_details]
        details_data = [tuple(useable_details_df.iloc[i]) for i in range(len(useable_details_df))]
        cursor.executemany("INSERT IGNORE INTO details VALUES ({})".format(("%s, "*len(columns_of_details))[:-2]), details_data)
        conn.commit()

    @classmethod
    def update_predict_points(cls, conn, cursor, ncodes, predict_points):
        """作品の予測ポイントを更新"""
        data = [(ncode, predict_point) for ncode, predict_point in zip(ncodes, predict_points)]
        cursor.executemany("UPDATE details SET predict_point=%s WHERE ncode=%s", data)
        conn.commit()

    @classmethod
    def get_details_df_iterator(cls, conn, test):
        if test:
            details_df_iterator = pd.read_sql_query("SELECT * FROM details LIMIT 64", conn, chunksize=cls.chunksize)
        else:
            details_df_iterator = pd.read_sql_query("SELECT * FROM details WHERE predict_point='Nan'", conn, chunksize=chunksize)
        return details_df_iterator


class ElasticsearchConnector(object):
    """Elasticsearchとの接続を担うクラス"""

    batch_size = ELASTICSEARCH_BATCH_SIZE
    host_name = ELASTICSEARCH_HOST_NAME
    feature_extraction_url = FEATURE_EXTRACTION_URL
    h_dim = H_DIM

    @classmethod
    def get_client(cls):
        return Elasticsearch(cls.host_name)

    @classmethod
    def create_indices(cls, client):
        """indexの新規作成"""
        mappings = {
            'properties': {
                'ncode': {'type': 'keyword'},
                'writer': {'type': 'text'},
                'keyword': {'type': 'text'}, 
                'story': {'type': 'text'},
                'feature': {'type': 'dense_vector', 'dims': cls.h_dim},
            }
        }
        client.indices.create(index='features', body={'mappings': mappings})

    @classmethod
    def add_details(cls, client, details_df):
        """作品の詳細情報の一部と特徴量を追加"""
        sub_df_iterator = cls.__generate_sub_df(details_df)
        for sub_df in sub_df_iterator:
            ncodes, texts, stories, keywords, writers = \
                list(sub_df.ncode), list(sub_df.text), list(sub_df.story), list(sub_df.keyword), list(sub_df.writer)
            
            features = BERTServerConnector.extract_features(texts)
            bulk(client, cls.__generate_es_data(ncodes, writers, keywords, stories, features))

    @classmethod
    def __generate_sub_df(cls, details_df):
        """DataFrameをミニバッチに分割するサブロジック"""
        recommendable_df = details_df[(details_df['predict_point'] == 1) & (details_df['global_point'] == 0)]
        if len(recommendable_df) != 0:
            for i in range(len(recommendable_df) // cls.batch_size + 1):
                start, end = i * cls.batch_size, (i + 1) * cls.batch_size
                sub_recommendable_df = recommendable_df.iloc[start:end]
                yield sub_recommendable_df

    @classmethod
    def __generate_es_data(
        cls, ncodes: List[str], writers: List[str], keywords: List[str], stories: List[str],  \
        features: List[float]) -> dict:
        """Elasticsearchへバルクインサートするための前処理"""
        for ncode, writer, keyword, story, feature in zip(ncodes, writers, keywords, stories, features):
            yield {
                '_index': 'features',
                'ncode': ncode,
                'writer': writer,
                'keyword': keyword,
                'story': story,
                'feature': feature,
            }


class BERTServerConnector(object):
    """BERTServerとの接続を担うクラス"""

    feature_extraction_url = FEATURE_EXTRACTION_URL

    @classmethod
    def extract_features(cls, texts: List[str]) -> List[float]:
        headers = {'Content-Type': 'application/json'}
        data = {'texts': texts}
        response = requests.get(cls.feature_extraction_url, headers=headers, json=data)
        features = response.json()['prediction']
        return features


class MLServerConnector(object):
    """MLServerとの接続を担うクラス"""

    point_prediction_url = POINT_PREDICTION_URL

    @classmethod
    def predict_point(cls, details_df: DataFrame) -> List[int]:
        details_df = cls.preprocessing(details_df)
        
        headers = {'Content-Type': 'application/json'}
        data = {}
        data = {column: list(details_df[column]) for column in list(details_df.columns)}
        data = json.dumps(data)
        response = requests.get(cls.point_prediction_url, headers=headers, json=data)

        predicted_points = response.json()['prediction']
        return predicted_points

    @classmethod
    def preprocessing(cls, details_df: DataFrame) -> DataFrame:
        '''ポイント予測のために新しい特徴量を作成する。
        
        Variables:
            title_length: タイトルの文字数
            story_length: あらすじの文字数
            text_length: 本文の文字数
            keyword_number: キーワードの数
            noun_proportion_in_text: 本文中における文字数当たりの名詞数
        '''
        mecab = MeCab.Tagger("-Ochasen")
        
        for column in ['title', 'story', 'text']:
            details_df[column + '_length'] = details_df[column].apply(lambda x: len(str(x)))
        details_df['keyword_number'] = details_df['keyword'].apply(lambda x: len(str(x).split(' ')))
        details_df['noun_proportion_in_text'] = details_df.text.apply(
                lambda x: cls.count_noun_number(mecab, str(x)) / len(str(x))
        )
        return details_df

    @classmethod
    def count_noun_number(cls, mecab: MeCab.Tagger, text: str) -> int:
        count = []
        for line in mecab.parse(str(text)).splitlines():
            try:
                if "名詞" in line.split()[-1]:
                    count.append(line)
            except:
                pass
        return len(set(count))