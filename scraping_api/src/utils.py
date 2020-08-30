import json
from typing import List, Tuple, Any

from bs4 import BeautifulSoup
import elasticsearch
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import MeCab
import MySQLdb
import pandas as pd
import requests


# DB utils
def get_connector_and_cursor(host_name: str) -> Tuple[Any, Any]:
    conn = MySQLdb.connect(
        host = host_name,
        port = 3306,
        user = 'root',
        password = 'root',
        database = 'maindb',
        use_unicode=True,
        charset="utf8"
    )
    cursor = conn.cursor()
    return conn, cursor


# Point prediction utils
def count_noun_number(mecab: MeCab.Tagger, text: str) -> int:
    count = []
    for line in mecab.parse(str(text)).splitlines():
        try:
            if "名詞" in line.split()[-1]:
                count.append(line)
        except:
            pass
    return len(set(count))


def preprocessing(detail_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    '''ポイント予測のために新しい特徴量を作成する。
    
    Fields:
        title_length: タイトルの文字数
        story_length: あらすじの文字数
        text_length: 本文の文字数
        keyword_number: キーワードの数
        noun_proportion_in_text: 本文中における文字数当たりの名詞数
    '''
    mecab = MeCab.Tagger("-Ochasen")
    
    for column in ['title', 'story', 'text']:
        detail_df[column + '_length'] = detail_df[column].apply(lambda x: len(str(x)))
    detail_df['keyword_number'] = detail_df['keyword'].apply(lambda x: len(str(x).split(' ')))
    detail_df['noun_proportion_in_text'] = detail_df.text.apply(
            lambda x: count_noun_number(mecab, str(x)) / len(str(x))
    )
    return detail_df


def point_prediction(url: str, detail_df: pd.core.frame.DataFrame) -> List[int]:
    detail_df = preprocessing(detail_df)
    
    headers = {'Content-Type': 'application/json'}
    data = {}
    data = {column: list(detail_df[column]) for column in list(detail_df.columns)}
    data = json.dumps(data)
    r_post = requests.post(url, headers=headers, json=data)

    predicted_points = r_post.json()['prediction']
    return predicted_points


# Feature extraction utils
def _generate_data(ncodes: List[str], features: List[float]) -> dict:
    for ncode, feature in zip(ncodes, features):
        yield {
            '_index': 'features',
            'ncode': ncode,
            'feature': feature
        }


def extract_features(url: str, texts: List[str]) -> List[float]:
    headers = {'Content-Type': 'application/json'}
    data = {'texts': texts}
    r_post = requests.post(url, headers=headers, json=data)
    features = r_post.json()['prediction']
    return features


def add_features_to_elasticsearch(client: elasticsearch.client.Elasticsearch, url: str, ncodes: List[str], texts: List[str], h_dim: int=64):
    features = extract_features(url, texts)
    
    if not client.indices.exists(index='features'):
        mappings = {
            'properties': {
                'ncode': {'type': 'keyword'},
                'feature': {'type': 'dense_vector', 'dims': h_dim}
            }
        }
        client.indices.create(index='features', body={'mappings': mappings })
    
    bulk(client, _generate_data(ncodes, features))