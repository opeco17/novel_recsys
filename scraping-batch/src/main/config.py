import os
import logging

from dotenv import load_dotenv


base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))
load_dotenv(abs_path_of('config.env'))


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 

    # Log
    LOG_FILE = abs_path_of('log/batch.log')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)

    # Parameter
    H_DIM = 64
    SCRAPING_DETAILS_BATCH_SIZE = 32
    ELASTICSEARCH_BATCH_SIZE = 16
    DB_BATCH_SIZE = 32
    INTERVAL = 0.1

    # Narou
    NAROU_URL = 'https://ncode.syosetu.com/'
    NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'

    # External server
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        DB_PORT = 30100
        ELASTICSEARCH_HOST_NAME = 'localhost:30101'
        FEATURE_EXTRACTION_URL = 'http://localhost:30002/predict'
        POINT_PREDICTION_URL = 'http://localhost:30003/predict'
        DB_HOST_NAME = '0.0.0.0'

    elif host == 'container':
        DB_PORT = 3306
        NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'
        ELASTICSEARCH_HOST_NAME = 'elasticsearch:9200'
        FEATURE_EXTRACTION_URL = 'http://bertserver:82/predict'
        POINT_PREDICTION_URL = 'http://mlserver:83/predict'
        DB_HOST_NAME = 'database'
