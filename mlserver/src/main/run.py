import json
import logging
import logging.handlers
import pickle

from flask import Flask

from config import Config


app = Flask(__name__)
app.config.from_object(Config)

app.logger.setLevel(app.config['LOG_LEVEL'])

model = pickle.load(open(app.config['LIGHTGBM_MODEL_PATH'], 'rb'))

with open(app.config['FEATURE_NAMES_PATH'], 'r') as f:
    feature_names = json.load(f)

from controller import *