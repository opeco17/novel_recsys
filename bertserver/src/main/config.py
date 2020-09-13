import logging
import os

from dotenv import load_dotenv

base_path = os.path.dirname(os.path.abspath(__file__))
abs_path_of = lambda path: os.path.normpath(os.path.join(base_path, path))
load_dotenv(abs_path_of('config.env'))


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess')

    # Log
    LOG_FILE = abs_path_of('log/batch.log')
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)

    # Parameter
    DB_BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    MAX_LENGTH = 512
    NUM_CLASSES = 21
    NUM_EPOCHS = 1
    H_DIM = 64
    TRAIN_BATCH_SIZE = 8

    # Path
    DATASET_DIR = abs_path_of('train/dataset')
    PARAMETER_DIR = abs_path_of('models/parameters')
    PARAMETER_PATH = abs_path_of('models/parameters/bert40000.pth')
    PRETRAINED_BERT_PATH = abs_path_of('models/parameters/pretrained_bert')
    PRETRAINED_TOKENIZER_PATH = abs_path_of('models/parameters/pretrained_tokenizer')
    TRAINED_MODEL_DIR = abs_path_of('train/trained_model')


    # External server
    DB_PORT = 3306
    host = os.environ.get('HOST', 'local')
    if host == 'local':
        DB_HOST_NAME = '0.0.0.0'

    elif host == 'container':
        DB_HOST_NAME = 'database'