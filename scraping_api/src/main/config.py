import os

from dotenv import load_dotenv


basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, 'config.env'))


class Config(object):
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 
    NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'

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
