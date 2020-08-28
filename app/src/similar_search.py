import requests
from elasticsearch import Elasticsearch



ELASTICSEARCH_HOST_NAME = 'elasticsearch'
FEATURE_EXTRACTION_URL = 'http://bertserver:3032/predict'
SCRAPING_TEXT_URL = 'http://scraping_api:3034/scraping_texts'

# ELASTICSEARCH_HOST_NAME = 'localhost:9200'
# FEATURE_EXTRACTION_URL = 'http://localhost:3032/predict'
# SCRAPING_TEXT_URL = 'http://localhost:3034/scraping_texts'



class SimilarTextSearch(object):
    
    def __init__(self):
        self.client = Elasticsearch(ELASTICSEARCH_HOST_NAME)
        self.feature_prediction_url = FEATURE_EXTRACTION_URL
    
    
    def similar_search_by_ncode(self, query_ncode):
        '''
        Args:
            str query_ncode: ncode query for similar search
        '''    
        if type(query_ncode) is not str:
            raise Exception('Argument query_ncode should be str.')

        query_to_search_query_ncode = {
            "query": {
                "term": {
                    "ncode": query_ncode
                }
            }
        }
        response = self.client.search(index='features', body=query_to_search_query_ncode)
        
        if len(response['hits']['hits']) == 0:
            query_text = self._scraping_text_by_ncode(query_ncode)
            recommend_ncodes = self.similar_search_by_text(query_text)
            
        else:
            query_feature = response['hits']['hits'][0]['_source']['feature']     
            recommend_ncodes = self._similar_search_by_feature(query_feature)
        
        return recommend_ncodes
    
    
    def similar_search_by_text(self, query_text):
        '''
        Args:
            str query_text: text query for similar search
        '''
        if type(query_text) is not list and type(query_text) is not str:
            raise Exception('Argument query_text should be list or str.')
        if type(query_text) is str:
            query_text = [query_text]
            
        query_feature = self._extract_feature(query_text)[0]
        
        recommend_ncodes = self._similar_search_by_feature(query_feature)
        return recommend_ncodes
            
        
    def _scraping_text_by_ncode(self, ncode):
        if type(ncode) is str:
            ncode = [ncode]
        headers = {'Content-Type': 'application/json'}
        data = {'ncodes': ncode}
        r_post = requests.post(SCRAPING_TEXT_URL, headers=headers, json=data)
        text = r_post.json()['texts']
        return text
        
    
    def _similar_search_by_feature(self, query_feature, recommend_num=10):
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
        
        response = self.client.search(index='features', body=query_for_similar_search)
        recommend_ncodes = []
        for i in range(min(recommend_num, len(response['hits']['hits']))):
            ncode = response['hits']['hits'][i]['_source']['ncode']
            recommend_ncodes.append(ncode)
        return recommend_ncodes
        
        
    def _extract_feature(self, text):
        '''
        Args:
            list<str> texts: texts of narou novel
        Return:
            list<float> features: feature vectors of item
        '''   
        headers = {'Content-Type': 'application/json'}
        data = {'texts': text}
        r_post = requests.post(FEATURE_EXTRACTION_URL, headers=headers, json=data)
        feature = r_post.json()['prediction']
        return feature