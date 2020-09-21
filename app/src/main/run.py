import logging
import logging.handlers

from flask import Flask, render_template

from config import Config


app = Flask(__name__)
app.config.from_object(Config)

app.logger.setLevel(app.config['LOG_LEVEL'])

from controller import *
