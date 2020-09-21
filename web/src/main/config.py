import json
import logging
import os

base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 

    # Mail
    with open(abs_path_of('mail_info.json'), 'r') as f:
        mail_info = json.load(f)
    MAIL_SERVER = mail_info['mail_server']
    MAIL_PORT = 465
    MAIL_USE_SSL = True
    MAIL_USERNAME = mail_info['mail_username']
    MAIL_PASSWORD = mail_info['mail_password']

    # Log
    LOG_FILE = abs_path_of('log/batch.log')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)

    # External server
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        print('Host is set to local.')
        NCODE_SEARCH_URL = 'http://localhost:3031/search_by_ncode'
        TEXT_SEARCH_URL = 'http://localhost:3031/search_by_text'

    elif host == 'container':
        print('Host is set to container.')
        NCODE_SEARCH_URL = 'http://app:3031/search_by_ncode'
        TEXT_SEARCH_URL = 'http://app:3031/search_by_text'
        