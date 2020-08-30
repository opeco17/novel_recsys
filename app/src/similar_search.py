from typing import List

from elasticsearch import Elasticsearch
import requests


ELASTICSEARCH_HOST_NAME = 'elasticsearch'
FEATURE_EXTRACTION_URL = 'http://bertserver:3032/predict'
SCRAPING_TEXT_URL = 'http://scraping_api:3034/scraping_texts'

# ELASTICSEARCH_HOST_NAME = 'localhost:9200'
# FEATURE_EXTRACTION_URL = 'http://localhost:3032/predict'
# SCRAPING_TEXT_URL = 'http://localhost:3034/scraping_texts'


class SimilarTextSearch(object):
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

    
    def similar_search_by_ncode(self, query_ncode: str) -> List[str]:

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
        
        recommend_ncodes = self.__similar_search_by_feature(query_feature)
        return recommend_ncodes
    
    def similar_search_by_text(self, query_text: List[str]) -> List[str]:
        query_feature = self.__extract_feature(query_text)
        recommend_ncodes = self.__similar_search_by_feature(query_feature)
        return recommend_ncodes    
        
    def __scraping_text_by_ncode(self, ncode: str) -> str:
        ncode = [ncode]
        headers = {'Content-Type': 'application/json'}
        data = {'ncodes': ncode}
        r_post = requests.post(self.scraping_text_url, headers=headers, json=data)
        text = r_post.json()['texts']
        return text
        
    def __similar_search_by_feature(self, query_feature: List[float], recommend_num: int=10) -> List[str]:

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
        recommend_ncodes = []
        for i in range(min(recommend_num, len(response))):
            ncode = response[i]['_source']['ncode']
            recommend_ncodes.append(ncode)
        return recommend_ncodes
        
    def __extract_feature(self, text: str) -> List[float]:
        headers = {'Content-Type': 'application/json'}
        data = {'texts': text}
        r_post = requests.post(self.feature_prediction_url, headers=headers, json=data)
        feature = r_post.json()['prediction'][0]
        return feature