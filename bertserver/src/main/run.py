import json
import logging
import logging.handlers
import os
import sys

from flask import Flask

from config import Config


app = Flask(__name__)
app.config.from_object(Config)

app.logger.setLevel(app.config['LOG_LEVEL'])

from controller import *