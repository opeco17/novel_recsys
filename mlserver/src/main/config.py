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

    # Path
    LIGHTGBM_MODEL_PATH = abs_path_of('models/ml_model/lightgbm_model.pkl')
    FEATURE_NAMES_PATH = abs_path_of('models/ml_model/feature_names.json')
