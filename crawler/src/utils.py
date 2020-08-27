import json

import requests
from bs4 import BeautifulSoup
import MySQLdb
import MeCab
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk



# DB utils

def get_connector_and_cursor(host_name):
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

def _count_noun_number(mecab, text):
    text = str(text)
    count = []
    for line in mecab.parse(text).splitlines():
        try:
            if "名詞" in line.split()[-1]:
                count.append(line)
        except:
            pass
    return len(set(count))


def _preprocessing(detail_df):
    '''
    New features: 
        title_length: length of title
        story_length: length of story
        text_length: length of text
        keyword_number: number of keywords
        noun_proportion_in_text: number of nouns in text per text length
    '''
    mecab = MeCab.Tagger("-Ochasen")
    
    for column in ['title', 'story', 'text']:
        detail_df[column + '_length'] = detail_df[column].apply(lambda x: len(str(x)))
    detail_df['keyword_number'] = detail_df['keyword'].apply(lambda x: len(str(x).split(' ')))
    detail_df['noun_proportion_in_text'] = detail_df.text.apply(lambda x: _count_noun_number(mecab, str(x)) / len(str(x)))
    return detail_df


def point_prediction(url, detail_df):
    '''
    Args:
        str url: url of point prediction api
        pandas.DataFrame detail_df: dataframe containing all features of item
    '''
    detail_df = _preprocessing(detail_df)
    
    headers = {'Content-Type': 'application/json'}
    data = {}
    data = {column: list(detail_df[column]) for column in list(detail_df.columns)}
    data = json.dumps(data)
    r_post = requests.post(url, headers=headers, json=data)

    predicted_points = r_post.json()['prediction']
    return predicted_points


# Feature extraction utils

def _generate_data(ncodes, features):
    for ncode, feature in zip(ncodes, features):
        yield {
            '_index': 'features',
            'ncode': ncode,
            'feature': feature
        }


def extract_features(url, texts):
    '''
    Args:
        str url: url of feature extraction api
        list<str> texts: texts of narou novel
    Return:
        list<float> features: feature vectors of item
    '''   
    headers = {'Content-Type': 'application/json'}
    data = {'texts': texts}
    r_post = requests.post(url, headers=headers, json=data)
    features = r_post.json()['prediction']
    return features


def register_features_to_elasticsearch(host, url, ncodes, texts, h_dim=64):
    '''
    Args: 
        str host: host name of elasticsearch
        str url: url of feature extraction api
        list<str> ncodes: ncodes to register
        texts<str> texts: texts to extract features
        h_dim: size of feature vector
    '''    

    features = extract_features(url, texts)
    
    client = Elasticsearch(host)
    
    mappings = {
        'properties': {
            'ncode': {'type': 'text'},
            'feature': {'type': 'dense_vector', 'dims': h_dim}
        }
    }
    
    if not client.indices.exists(index='features'):
        client.indices.create(index='features', body={ 'mappings': mappings })
    
    bulk(client, _generate_data(ncodes, features))