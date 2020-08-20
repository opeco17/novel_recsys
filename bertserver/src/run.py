import os
import json
from flask import Flask

from config import Config
from bert import load_model


app = Flask(__name__)
app.config.from_object(Config)

model = load_model()

from routes import *