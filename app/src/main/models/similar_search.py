import sys
from typing import List, Tuple, Dict
sys.path.append('..')

from elasticsearch import Elasticsearch
import requests

from config import Config
from models.similar_search_utils import TextScraper, ElasticsearchConnector, BERTServerConnector
from run import app


class SimilarItemSearch(object):
    """類似文書検索クラス

    ncodeまたはtextをクエリとして類似文書検索を行う。
    ncodeで指定した文書がElasticsearchに存在しない場合はScraping API経由で文書を取得する。

    Methods:
        similar_search_by_ncode: ncodeをクエリとして類似文書検索を行い対象のncodeを返す
        similar_search_by_text: textをクエリとして類似文書検索を行い対象のncodeを返す
        __extract_feature: 引数のtextの特徴量を抽出して返す
    """
    
    def __init__(self):
        self.client = ElasticsearchConnector.get_client()

    def similar_search_by_ncode(self, query_ncode: str) -> List[Dict]:
        query_feature = self.__get_feature_by_ncode(query_ncode)
        if query_feature == None:
            query_text = self.__scraping_text_by_ncode(query_ncode)
            query_feature = self.__extract_feature(query_text)
        recommend_list = self.__similar_search_by_feature(query_feature)
        return recommend_list
    
    def similar_search_by_text(self, query_text: List[str]) -> List[Dict]:
        query_feature = self.__extract_feature(query_text)
        recommend_list = self.__similar_search_by_feature(query_feature)
        return recommend_list

    def __get_feature_by_ncode(self, query_ncode: str) -> List[float]:
        query_feature = ElasticsearchConnector.get_feature_by_ncode(self.client, query_ncode)
        return query_feature
        
    def __scraping_text_by_ncode(self, query_ncode: str) -> str:
        query_text = TextScraper.scraping_text(query_ncode)
        return query_text
        
    def __similar_search_by_feature(self, query_feature: List[float]) -> List[Dict]:        
        recommend_list = ElasticsearchConnector.get_recommends_by_feature(self.client, query_feature)
        return recommend_list

    def __extract_feature(self, text: str) -> List[float]:
        feature = BERTServerConnector.extract_feature(text)
        return feature

    def __del__(self):
        self.client.close()