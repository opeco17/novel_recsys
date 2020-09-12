import os
import logging

from dotenv import load_dotenv


basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, 'config.env'))


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 

    # Log
    LOG_FILE = 'log/batch.log'
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)

    # Parameter
    H_DIM = 64
    SCRAPING_DETAILS_BATCH_SIZE = 32
    ELASTICSEARCH_BATCH_SIZE = 16
    DB_BATCH_SIZE = 32
    DB_PORT = 3306
    INTERVAL = 0.1

    # Narou
    NAROU_URL = 'https://ncode.syosetu.com/'
    NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'

    # External server
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        ELASTICSEARCH_HOST_NAME = 'localhost:9200'
        FEATURE_EXTRACTION_URL = 'http://localhost:3032/predict'
        POINT_PREDICTION_URL = 'http://localhost:3033/predict'
        DB_HOST_NAME = '0.0.0.0'

    elif host == 'container':
        NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'
        ELASTICSEARCH_HOST_NAME = 'elasticsearch'
        FEATURE_EXTRACTION_URL = 'http://bertserver:3032/predict'
        POINT_PREDICTION_URL = 'http://mlserver:3033/predict'
        DB_HOST_NAME = 'database'
