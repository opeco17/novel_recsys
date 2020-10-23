import json
import logging
import os

base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 
    
    # Parameter
    RECOMMEND_NUM = 25
    
    # Pagenation
    PAGENATION_NUM = 5
    PAGENATION_MESSAGE = '表示範囲 <b>{start}件 - {end}件 </b> 合計：<b>{total}</b>件'

    # Mail
    with open(abs_path_of('mail_info.json'), 'r') as mail_file:
        MAIL_INFO = json.load(mail_file)
    MAIL_SERVER = MAIL_INFO['mail_server']
    MAIL_PORT = 465
    MAIL_USE_SSL = True
    MAIL_USERNAME = MAIL_INFO['mail_username']
    MAIL_PASSWORD = MAIL_INFO['mail_password']
    
    # Admin
    with open(abs_path_of('admin_info.json'), 'r') as admin_file:
        ADMIN_INFO = json.load(admin_file)
        
    # Log
    LOG_FILE = abs_path_of('log/batch.log')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)

    # External server
    KIBANA_URL = 'http://localhost:30102'
    GRAFANA_URL = 'http://localhost:30103'
    
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        print('Host is set to local.')
        NCODE_SEARCH_URL = 'http://localhost:30001/search_by_ncode'
        TEXT_SEARCH_URL = 'http://localhost:30001/search_by_text'

    elif host == 'container':
        print('Host is set to container.')
        NCODE_SEARCH_URL = 'http://app/search_by_ncode'
        TEXT_SEARCH_URL = 'http://app/search_by_text'
