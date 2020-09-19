import json
import sys
from typing import Dict, Generator, List, Tuple, Any
sys.path.append('..')

from bs4 import BeautifulSoup
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import MeCab
import MySQLdb
from  MySQLdb.connections import Connection 
from MySQLdb.cursors import Cursor
import pandas as pd
from pandas.core.frame import DataFrame
import requests

from run import app
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
    def get_conn_and_cursor(cls) -> Tuple[Connection, Cursor]:
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
    def add_details(cls, conn: Connection, cursor: Cursor, details_df: DataFrame):
        """作品の詳細情報全てを追加"""
        cursor.execute("SHOW columns FROM details")
        columns_of_details = [column[0] for column in cursor.fetchall()]
        useable_details_df = details_df[columns_of_details]
        details_data = [tuple(useable_details_df.iloc[i]) for i in range(len(useable_details_df))]
        cursor.executemany("INSERT IGNORE INTO details VALUES ({})".format(("%s, "*len(columns_of_details))[:-2]), details_data)
        conn.commit()

    @classmethod
    def update_predict_points(cls, conn: Connection, cursor: Cursor, ncodes: List[str], predict_points: List[int]):
        """作品の予測ポイントを更新"""
        data = [(ncode, predict_point) for ncode, predict_point in zip(ncodes, predict_points)]
        cursor.executemany("UPDATE details SET predict_point=%s WHERE ncode=%s", data)
        conn.commit()

    @classmethod
    def get_details_df_iterator(cls, conn: Connection, test: bool, epoch: int) -> Generator:
        if test:
            details_df_iterator = pd.read_sql_query(f"SELECT * FROM details LIMIT {epoch * 64}", conn, chunksize=cls.chunksize)
        else:
            details_df_iterator = pd.read_sql_query("SELECT * FROM details WHERE predict_point='Nan'", conn, chunksize=cls.chunksize)
        return details_df_iterator


class ElasticsearchConnector(object):
    """Elasticsearchとの接続を担うクラス"""

    batch_size = ELASTICSEARCH_BATCH_SIZE
    host_name = ELASTICSEARCH_HOST_NAME
    feature_extraction_url = FEATURE_EXTRACTION_URL
    h_dim = H_DIM

    @classmethod
    def get_client(cls) -> Elasticsearch:
        return Elasticsearch(cls.host_name)

    @classmethod
    def create_indices(cls, client: Elasticsearch):
        """indexの新規作成"""
        mappings = {
            'properties': {
                'ncode': {'type': 'keyword'},
                'title': {'type': 'text'},
                'writer': {'type': 'text'},
                'keyword': {'type': 'text'}, 
                'story': {'type': 'text'},
                'genre': {'type': 'integer'},
                'biggenre': {'type': 'integer'},
                'feature': {'type': 'dense_vector', 'dims': cls.h_dim},
            }
        }
        client.indices.create(index='features', body={'mappings': mappings})

    @classmethod
    def add_details(cls, client: Elasticsearch, details_df: DataFrame):
        """作品の詳細情報の一部と特徴量を追加"""
        sub_df_iterator = cls.__generate_sub_df(details_df)
        for sub_df in sub_df_iterator:
            ncodes, titles, texts, stories, keywords, writers, genres, biggenres = \
                list(sub_df.ncode), list(sub_df.title), list(sub_df.text), list(sub_df.story), \
                list(sub_df.keyword), list(sub_df.writer), list(sub_df.genre), list(sub_df.biggenre)
            features = BERTServerConnector.extract_features(texts)
            bulk(client, cls.__generate_es_data(ncodes, titles, writers, keywords, stories, genres, biggenres, features))

    @classmethod
    def __generate_sub_df(cls, details_df: DataFrame) -> DataFrame:
        """DataFrameをミニバッチに分割するサブロジック"""
        recommendable_df = details_df[(details_df['predict_point'] == 1) & (details_df['global_point'] == 0)]
        if len(recommendable_df) != 0:
            for i in range(len(recommendable_df) // cls.batch_size + 1):
                start, end = i * cls.batch_size, (i + 1) * cls.batch_size
                sub_recommendable_df = recommendable_df.iloc[start:end]
                yield sub_recommendable_df

    @classmethod
    def __generate_es_data(cls, ncodes: List[str], titles: List[str], writers: List[str], keywords: List[str], stories: List[str], \
                            genres: List[int], biggenres: List[int], features: List[float]) -> Dict:
        """Elasticsearchへバルクインサートするための前処理"""
        for ncode, title, writer, keyword, story, genre, biggenre, feature in zip(ncodes, titles, writers, keywords, stories, genres, biggenres, features):
            yield {
                '_index': 'features',
                'ncode': ncode,
                'title': title,
                'writer': writer,
                'keyword': keyword,
                'story': story,
                'genre': genre, 
                'biggenres': biggenre,
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
        headers = {'Content-Type': 'application/json'}
        data = {}
        data = {column: list(details_df[column]) for column in list(details_df.columns)}
        data = json.dumps(data)
        response = requests.get(cls.point_prediction_url, headers=headers, json=data)
        app.logger.info(f"Point predicted: {response}")
        predicted_points = response.json()['prediction']
        return predicted_points