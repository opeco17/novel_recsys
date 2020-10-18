from typing import Dict, Iterator

import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pandas.core.frame import DataFrame

from bertserver_connector import BERTServerConnector
from config import Config
from logger import logger


class ElasticsearchConnector(object):
    """Elasticsearchとの接続を担うクラス"""

    @classmethod
    def get_client(cls) -> Elasticsearch:
        try:
            return Elasticsearch(Config.ELASTICSEARCH_HOST_NAME)
        except Exception as e:
            extra = {'Class': 'ElasticsearchConnector', 'Method': 'get_client', 'ErrorType': type(e), 'Error': str(e)}
            logger.error('Unable to get elasticsearch client.')
            raise

    @classmethod
    def create_details_index_if_not_exist(cls, client: Elasticsearch) -> None:
        """indexの新規作成"""
        if client.indices.exists(index='details'):
            return None
        
        mappings = {'properties': Config.ES_DETAILS_SCHEMA}
        try:
            client.indices.create(index='details', body={'mappings': mappings})
        except Exception as e:
            # Podが同時にindexを作成しようとするとエラーが発生するのでここで止める
            extra = {'Class': 'ElasticsearchConnector', 'Method': 'create_details_index', 'ErrorType': type(e), 'Error': str(e)}
            logger.error('Unable to create details index.')

    @classmethod
    def insert_details(cls, client: Elasticsearch, details_df: DataFrame) -> Dict:
        """作品の詳細情報の一部と特徴量を追加"""
        try:
            bulk(client, cls.__generate_es_data(details_df))
            logger.info(f"{len(details_df)} data was inserted to elasticsearch.")
        except Exception as e:
            extra = {'Class': 'ElasticsearchConnector', 'Method': 'insert_details', 'ErrorType': type(e), 'Error': str(e)}
            logger.error('Unable to insert details to elasticsearch.')
            raise
        
    @classmethod
    def __generate_es_data(cls, details_df: DataFrame) -> Iterator[Dict]:
        for ncode, title, writer, keyword, story, genre, biggenre, feature in \
            zip(details_df.ncode, details_df.title, details_df.writer, details_df.keyword, \
                details_df.story, details_df.genre, details_df.biggenre, details_df.feature):
                
            yield {
                '_index': 'details',
                'ncode': ncode,
                'title': title,
                'writer': writer,
                'keyword': keyword,
                'story': story,
                'genre': genre, 
                'biggenre': biggenre,
                'feature': feature,
            }        