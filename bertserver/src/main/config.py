import os

from dotenv import load_dotenv


base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))
load_dotenv(abs_path_of('config.env'))


class Config(object):
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess')
    H_DIM = 64
    MAX_LENGTH = 512
    PARAMETER_PATH = abs_path_of('parameters/bert40000.pth')
    PRETRAINED_BERT_PATH = abs_path_of('parameters/pretrained_bert')
    PRETRAINED_TOKENIZER_PATH = abs_path_of('parameters/pretrained_tokenizer')
    
