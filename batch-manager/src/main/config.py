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
    
    # Target
    TARGET = os.environ.get('TARGET', 'Batch')
        
    # Log
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)
    
    # Pararell
    COMPLETIONS = int(os.environ.get('COMPLETIONS', 2))
    
    # Webhook
    WEBHOOK_URL = os.environ.get('WEBHOOK_URL')