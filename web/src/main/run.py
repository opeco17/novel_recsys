import json
import logging
import logging.handlers

from flask import Flask, render_template
from flask.logging import default_handler
from flask_bootstrap import Bootstrap
from flask_httpauth import HTTPDigestAuth
from flask_mail import Mail
import json_log_formatter

from config import Config
from logger import get_json_stream_handler


# Application
app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

# Logging
app.logger.removeHandler(default_handler)
app.logger.addHandler(get_json_stream_handler())
app.logger.setLevel(app.config['LOG_LEVEL'])

# Mail
mail = Mail()
mail.init_app(app)

# Auth
auth = HTTPDigestAuth()

from controller import *