from flask import Flask
from flask.logging import default_handler

from config import Config


# Application
app = Flask(__name__)
app.config.from_object(Config)

# Logging
app.logger.removeHandler(default_handler)

from controller import *