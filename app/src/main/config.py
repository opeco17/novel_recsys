import logging
import os

base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 

    # Parameter
    RECOMMEND_NUM = 10

    # Log
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)

    # Narou
    NAROU_URL = 'https://ncode.syosetu.com/'
    NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'

    # External server
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        print('Host is set to local.')
        ELASTICSEARCH_HOST_NAME = 'localhost:30101'
        FEATURE_EXTRACTION_URL = 'http://localhost:30002/predict'

    elif host == 'container':
        print('Host is set to container.')
        ELASTICSEARCH_HOST_NAME = 'elasticsearch'
        FEATURE_EXTRACTION_URL = 'http://bertserver/predict'