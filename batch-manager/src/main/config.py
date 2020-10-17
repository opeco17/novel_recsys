import logging
import os


class Config(object):
    # Basic
    JSON_AS_ASCII = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess') 
        
    # Log
    LOG_LEVEL = os.environ.get('LOG_LEVEL', logging.DEBUG)
    
    # Pararell
    COMPLETIONS = int(os.environ.get('COMPLETIONS', 2))