import os
import logging

from dotenv import load_dotenv


base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))
load_dotenv(abs_path_of('config.env'))


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    APP_NAME = os.environ.get('APP_NAME', 'scraping-batch')

    # Log
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)
    
    # DB Schema
    DEFAULT_NULL_COLUMNS = ['predicted_point']

    # Parameter
    ITERATION_OF_TEST = 2
    SCRAPING_INTERVAL = 0.5
    SCRAPING_DETAILS_BATCH_SIZE = 32
    
    # Webhook
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL')

    # Narou
    NAROU_URL = 'https://ncode.syosetu.com/'
    NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'

    # External server
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER_NAME = os.environ.get('DB_USER_NAME')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')
    
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        DB_PORT = 30100
        DB_HOST_NAME = '0.0.0.0'

    elif host == 'container':
        DB_PORT = 3306
        DB_HOST_NAME = 'database'