from typing import List, Tuple, Dict

from elasticsearch import Elasticsearch
import requests

from config import Config


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

        query_to_pull_query_feature_from_es = {
            "query": {
                "term": {
                    "ncode": query_ncode
                }
            }
        }
        response = self.client.search(index='features', body=query_to_pull_query_feature_from_es)['hits']['hits']
        query_feature_is_in_es = len(response) != 0

        if query_feature_is_in_es:
            query_feature = response[0]['_source']['feature']     
        else:
            query_text = self.__scraping_text_by_ncode(query_ncode)
            query_feature = self.__extract_feature(query_text)
        
        recommend_list = self.__similar_search_by_feature(query_feature)
        return recommend_list
    
    def similar_search_by_text(self, query_text: List[str]) -> List[Dict]:
        query_feature = self.__extract_feature(query_text)
        recommend_list = self.__similar_search_by_feature(query_feature)
        return recommend_list
        
    def __scraping_text_by_ncode(self, ncode: str) -> str:
        ncode = [ncode]
        headers = {'Content-Type': 'application/json'}
        data = {'ncodes': ncode}
        r_post = requests.post(self.scraping_text_url, headers=headers, json=data)
        text = r_post.json()['texts']
        return text
        
    def __similar_search_by_feature(self, query_feature: List[float], recommend_num: int=10) -> List[Dict]:

        query_for_similar_search = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vec, doc['feature']) + 1.0", # Elasticsearch does not allow negative scores
                        "params": {
                            "query_vec": query_feature
                        }
                    }
                }
            }
        }
        
        response = self.client.search(index='features', body=query_for_similar_search)['hits']['hits']
        recommend_list = []
        for i in range(min(recommend_num, len(response))):
            recommend_data = response[i]['_source']
            recommend_data.pop('feature')
            recommend_list.append(recommend_data)
        return recommend_list
        
    def __extract_feature(self, text: str) -> List[float]:
        headers = {'Content-Type': 'application/json'}
        data = {'texts': text}
        r_post = requests.post(self.feature_prediction_url, headers=headers, json=data)
        feature = r_post.json()['prediction'][0]
        return feature