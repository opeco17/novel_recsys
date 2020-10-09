import json
import pickle

from flask import Flask
from flask.logging import default_handler

from config import Config
from logger import get_json_stream_handler


# Application
app = Flask(__name__)
app.config.from_object(Config)

# Logging
app.logger.removeHandler(default_handler)
app.logger.addHandler(get_json_stream_handler())
app.logger.setLevel(app.config['LOG_LEVEL'])

# Machine Learning
model = pickle.load(open(app.config['LIGHTGBM_MODEL_PATH'], 'rb'))
with open(app.config['FEATURE_NAMES_PATH'], 'r') as f:
    feature_names = json.load(f)

from controller import *