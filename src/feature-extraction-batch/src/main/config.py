import json
import os
import logging

from dotenv import load_dotenv


base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))
load_dotenv(abs_path_of('config.env'))


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    APP_NAME = os.environ.get('APP_NAME', 'feature-extraction-batch')
    
    # Parameter
    FEATURE_EXTRACTION_BATCH_SIZE = 16

    # Webhook
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL')
    
    # Log
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)
    
    # Elasticsearch
    with open(abs_path_of('./es_details_schema.json')) as f:
        ES_DETAILS_SCHEMA = json.load(f)

    # External server
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER_NAME = os.environ.get('DB_USER_NAME')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')
    
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        DB_PORT = 30100
        DB_HOST_NAME = '0.0.0.0'
        ELASTICSEARCH_HOST_NAME = 'localhost:30101'
        FEATURE_EXTRACTION_URL = 'http://localhost:30002/predict'
        POP_DATA_URL = 'http://localhost:30005/pop_data'
        FAIL_DATA_URL = 'http://localhost:30005/fail_data'
        SUCCESS_DATA_URL = 'http://localhost:30005/success_data'
        
    elif host == 'container':
        DB_PORT = 3306
        DB_HOST_NAME = 'database'
        ELASTICSEARCH_HOST_NAME = 'elasticsearch:9200'
        FEATURE_EXTRACTION_URL = 'http://bertserver/predict'
        POP_DATA_URL = 'http://feature-extraction-manager:3035/pop_data'
        FAIL_DATA_URL = 'http://feature-extraction-manager:3035/fail_data'
        SUCCESS_DATA_URL = 'http://feature-extraction-manager:3035/success_data'