import sys
from urllib.request import urlopen
sys.path.append('..')

from bs4 import BeautifulSoup


class TextScraper(object):
    """小説家になろうAPIから本文をスクレイピングするためのクラス"""

    @classmethod
    def scraping_text(cls, ncode: str) -> str:
        """ncodeをクエリとして本文のスクレイピングを実行する"""

        base_url = 'https://ncode.syosetu.com/' + ncode
        text = None
        c = 0
        while c < 5:
            try:
                bs_obj = cls.__make_bs_obj(base_url + '/')
                if bs_obj.findAll("dl", {"class": "novel_sublist2"}): # 連載作品の場合
                    bs_obj = cls.__make_bs_obj(base_url + '/1/')
                text = cls.__get_text(bs_obj)  
                break
            except Exception as e:
                print(e)
                c += 1 
        return text


    @classmethod
    def __make_bs_obj(cls, url: str) -> BeautifulSoup:
        html = urlopen(url)
        return BeautifulSoup(html, 'html.parser')

    @classmethod
    def __get_text(cls, bs_obj: BeautifulSoup) -> str:
        text = ""
        text_htmls = bs_obj.findAll('div', {'id': 'novel_honbun'})[0].findAll('p')
        for text_html in text_htmls:
            text = text + text_html.get_text() + "\n"
        return text


class ElasticsearchConnector(object):
    """Elasticsearchへの接続を行うためのクラス"""

    @classmethod
    def get_feature_by_ncode(cls, client, ncode):
        """ncodeをクエリとしてElasticsearchから特徴量を抽出"""
        query = {
            "query": {
                "term": {
                    "ncode": ncode
                }
            }
        }
        response = client.search(index='features', body=query)['hits']['hits']
        if len(response) != 0:
            query_feature = response[0]['_source']['feature']    
        else:
            query_feature = None
        return query_feature


    @classmethod
    def get_recommends_by_feature(cls, client, feature, recommend_num):
        """特徴量をクエリとしてElasticsearchから類似作品のレコメンドリストを抽出"""
        
        query_for_similar_search = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vec, doc['feature']) + 1.0", # Elasticsearch does not allow negative scores
                        "params": {
                            "query_vec": feature
                        }
                    }
                }
            }
        }
        response = client.search(index='features', body=query_for_similar_search)['hits']['hits']
        recommend_list = []
        for i in range(min(recommend_num, len(response))):
            recommend_data = response[i]['_source']
            recommend_data.pop('feature')
            recommend_list.append(recommend_data)
        return recommend_list