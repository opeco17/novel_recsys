import os
import json
import pickle
from flask import Flask

from config import Config


MODEL_PATH = './app/model/lightgbm_model.pkl'
FEATURE_NAMES_PATH = './app/model/feature_names.json'


app = Flask(__name__)
app.config.from_object(Config)

model = pickle.load(open(MODEL_PATH, 'rb'))

with open(FEATURE_NAMES_PATH, 'r') as f:
    feature_names = json.load(f)

from app import routes
