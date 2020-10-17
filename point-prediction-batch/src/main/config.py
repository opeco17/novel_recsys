import os
import logging

from dotenv import load_dotenv


base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))
load_dotenv(abs_path_of('config.env'))


class Config(object):
    # Basic
    JSON_AS_ASCII = False

    # Log
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)

    # External server
    DB_NAME = os.environ.get('DB_NAME')
    DB_USER_NAME = os.environ.get('DB_USER_NAME')
    DB_PASSWORD = os.environ.get('DB_PASSWORD')
    
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        DB_PORT = 30100
        DB_HOST_NAME = '0.0.0.0'
        POP_DATA_URL = 'http://localhost:30004/pop_data'
        DELETE_DATA_URL = 'http://localhost:30004/delete_data'

    elif host == 'container':
        DB_PORT = 3306
        DB_HOST_NAME = 'database'
        POP_DATA_URL = 'http://:ml-batch-manager:3034/pop_data'
        DELETE_DATA_URL = 'http://:ml-batch-manager:3034/delete_data'