import json

import requests
import numpy as np
import pandas as pd

from utils import *


TEST = True
MINIBATCH_SIZE = 10
NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'
ELASTICSEARCH_HOST_NAME = 'elasticsearch'
FEATURE_EXTRACTION_URL = 'http://bertserver:3032/predict'
POINT_PREDICTION_URL = 'http://mlserver:3033/predict'
DB_HOST_NAME = 'db'
# ELASTICSEARCH_HOST_NAME = 'localhost:9200'
# FEATURE_EXTRACTION_URL = 'http://localhost:3032/predict'
# POINT_PREDICTION_URL = 'http://localhost:3033/predict'
# DB_HOST_NAME = '0.0.0.0'


def register_all_data():
    conn, cursor = get_connector_and_cursor(DB_HOST_NAME)
    if TEST:
        detail_df = pd.read_sql_query("SELECT * FROM details LIMIT 10", conn)
    else:
        detail_df = pd.read_sql_query("SELECT * FROM details WHERE predict_point='Nan'", conn)

    predicted_point = point_prediction(POINT_PREDICTION_URL, detail_df)
    detail_df['predict_point'] = predicted_point
    target_detail_df = detail_df[(detail_df['predict_point']==1) & (detail_df['global_point']==0)]
    if len(target_detail_df) != 0:      
        ncodes = list(target_detail_df.ncode)
        texts = list(target_detail_df.text)

        for i in range(len(ncodes) // MINIBATCH_SIZE + 1):
            register_features_to_elasticsearch(ELASTICSEARCH_HOST_NAME, FEATURE_EXTRACTION_URL, ncodes[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE], texts[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE])
            if TEST:
                break

    print('{} data is inserted to Elasticsearch.'.format(len(target_detail_df)))
    conn.commit()
    conn.close()


if __name__ == '__main__':
    register_all_data()