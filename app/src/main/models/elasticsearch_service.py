import sys
sys.path.append('..')


def get_feature_by_ncode(client, ncode):
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


def get_recommends_by_feature(client, feature, recommend_num):
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