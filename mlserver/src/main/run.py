import json
import pickle

from flask import Flask
from flask.logging import default_handler

from config import Config

# Application
app = Flask(__name__)
app.config.from_object(Config)

# Logging
app.logger.removeHandler(default_handler)

# Machine Learning
model = pickle.load(open(app.config['LIGHTGBM_MODEL_PATH'], 'rb'))
with open(app.config['FEATURE_NAMES_PATH'], 'r') as f:
    feature_names = json.load(f)

from controller import *