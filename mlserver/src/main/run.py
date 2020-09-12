import json
import logging
import logging.handlers
import pickle

from flask import Flask

from config import Config


app = Flask(__name__)
app.config.from_object(Config)

handler = logging.handlers.RotatingFileHandler(app.config['LOG_FILE'], "a+", maxBytes=3000, backupCount=5)
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')) 
app.logger.addHandler(handler)
app.logger.setLevel(app.config['LOG_LEVEL'])

model = pickle.load(open(app.config['LIGHTGBM_MODEL_PATH'], 'rb'))

with open(app.config['FEATURE_NAMES_PATH'], 'r') as f:
    feature_names = json.load(f)

from controller import *