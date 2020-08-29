import os

from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, 'config.env'))


class Config(object):
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    DB_PATH = os.environ.get('DB_PATH')
    DETAILS_TABLE_NAME = os.environ.get('DETAILS_TABLE_NAME')
    DB_EXTENSION_PATH = os.environ.get('DB_EXTENSION_PATH')
    
