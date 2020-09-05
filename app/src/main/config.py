import os


class Config(object):
    DEBUG = False 
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 

    host = os.environ.get('HOST', 'local')
    if host == 'local':
        print('Host is set to local.')
        ELASTICSEARCH_HOST_NAME = 'localhost:9200'
        FEATURE_EXTRACTION_URL = 'http://localhost:3032/predict'
        SCRAPING_TEXT_URL = 'http://localhost:3034/scraping_texts'

    elif host == 'container':
        print('Host is set to container.')
        ELASTICSEARCH_HOST_NAME = 'elasticsearch'
        FEATURE_EXTRACTION_URL = 'http://bertserver:3032/predict'
        SCRAPING_TEXT_URL = 'http://scraping_api:3034/scraping_texts'