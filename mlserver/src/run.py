import os
import json
import pickle
from logging.handlers import RotatingFileHandler

from flask import Flask

from config import Config


app = Flask(__name__)
app.config.from_object(Config)

model = pickle.load(open(app.config['LIGHTGBM_MODEL_PATH'], 'rb'))

with open(app.config['FEATURE_NAMES_PATH'], 'r') as f:
    feature_names = json.load(f)

from routes import *