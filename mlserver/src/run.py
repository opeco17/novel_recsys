import os
import json
import pickle
from flask import Flask

from config import Config
from bert import load_model


MODEL_PATH = './model/lightgbm_model.pkl'

app = Flask(__name__)
app.config.from_object(Config)

model = pickle.load(open(MODEL_PATH, 'rb'))

from routes import *