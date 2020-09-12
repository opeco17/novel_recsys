import logging
import os


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 

    # Parameter
    RECOMMEND_NUM = 10

    # Log
    LOG_FILE = 'log/batch.log'
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)

    # Narou
    NAROU_URL = 'https://ncode.syosetu.com/'
    NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'

    # External server
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        print('Host is set to local.')
        ELASTICSEARCH_HOST_NAME = 'localhost:9200'
        FEATURE_EXTRACTION_URL = 'http://localhost:3032/predict'

    elif host == 'container':
        print('Host is set to container.')
        ELASTICSEARCH_HOST_NAME = 'elasticsearch'
        FEATURE_EXTRACTION_URL = 'http://bertserver:3032/predict'