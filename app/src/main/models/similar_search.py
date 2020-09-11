import sys
from typing import List, Tuple, Dict
sys.path.append('..')

from elasticsearch import Elasticsearch
import requests

from config import Config
from models.scraping_text import scraping_text
from models.elasticsearch_service import *


ELASTICSEARCH_HOST_NAME = Config.ELASTICSEARCH_HOST_NAME
FEATURE_EXTRACTION_URL = Config.FEATURE_EXTRACTION_URL 
SCRAPING_TEXT_URL = Config.SCRAPING_TEXT_URL


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
        self.elasticsearch_host_name = ELASTICSEARCH_HOST_NAME
        self.feature_prediction_url = FEATURE_EXTRACTION_URL
        self.scraping_text_url = SCRAPING_TEXT_URL
        self.client = Elasticsearch(self.elasticsearch_host_name)

    def similar_search_by_ncode(self, query_ncode: str) -> List[Dict]:
        query_feature = get_feature_by_ncode(self.client, query_ncode)
        if query_feature == None:
            query_text = self.__scraping_text_by_ncode(query_ncode)
            query_feature = self.__extract_feature(query_text)
        recommend_list = self.__similar_search_by_feature(query_feature)
        return recommend_list
    
    def similar_search_by_text(self, query_text: List[str]) -> List[Dict]:
        query_feature = self.__extract_feature(query_text)
        recommend_list = self.__similar_search_by_feature(query_feature)
        return recommend_list
        
    def __scraping_text_by_ncode(self, ncode: str) -> str:
        text = scraping_text(ncode)
        return text
        
    def __similar_search_by_feature(self, query_feature: List[float], recommend_num: int=10) -> List[Dict]:        
        recommend_list = get_recommends_by_feature(self.client, query_feature, recommend_num)
        return recommend_list

    def __extract_feature(self, text: str) -> List[float]:
        headers = {'Content-Type': 'application/json'}
        data = {'texts': text}
        r_post = requests.post(self.feature_prediction_url, headers=headers, json=data)
        feature = r_post.json()['prediction'][0]
        return feature