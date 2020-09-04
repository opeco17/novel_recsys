import os

from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, 'config.env'))


class Config(object):
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess')

    H_DIM = 64
    MAX_LENGTH = 512
    PARAMETER_PATH = 'parameters/bert40000.pth'
    PRETRAINED_BERT_PATH = 'parameters/pretrained_bert'
    PRETRAINED_TOKENIZER_PATH = 'parameters/pretrained_tokenizer'
    
