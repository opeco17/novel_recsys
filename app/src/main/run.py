import os
import json

from flask import Flask, render_template
from flask_bootstrap import Bootstrap

from config import Config
from utils import ResponseMakerForNcodeAndText


app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

ncode_response_maker = ResponseMakerForNcodeAndText('ncode')
text_response_maker = ResponseMakerForNcodeAndText('text')

from controller import *
