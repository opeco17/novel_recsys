import os

from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, 'config.env'))


class Config(object):
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
