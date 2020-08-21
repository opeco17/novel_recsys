import os
import json
from flask import Flask

from config import Config
from bert import load_model


app = Flask(__name__)
app.config.from_object(Config)

model = load_model(
    app.config['H_DIM'],
    app.config['MAX_LENGTH'],
    app.config['PARAMETER_PATH'],
    app.config['PRETRAINED_BERT_PATH'],
    app.config['PRETRAINED_TOKENIZER_PATH']
)

from routes import *