import os
import json

from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from config import Config
from similar_search import SimilarTextSearch


app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

similar_text_search = SimilarTextSearch()

from routes import *