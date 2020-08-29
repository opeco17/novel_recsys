import os

from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, 'config.env'))


class Config(object):
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY')
    H_DIM = int(os.environ.get('H_DIM'))
    MAX_LENGTH = int(os.environ.get('MAX_LENGTH'))
    PARAMETER_PATH = os.environ.get('PARAMETER_PATH')
    PRETRAINED_BERT_PATH = os.environ.get('PRETRAINED_BERT_PATH')
    PRETRAINED_TOKENIZER_PATH = os.environ.get('PRETRAINED_TOKENIZER_PATH')
    
