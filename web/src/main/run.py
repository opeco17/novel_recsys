import json
import logging
import logging.handlers

from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_httpauth import HTTPDigestAuth
from flask_mail import Mail

from config import Config


# Application
app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

app.logger.setLevel(app.config['LOG_LEVEL'])

# Mail
mail = Mail()
mail.init_app(app)

# Auth
auth = HTTPDigestAuth()

from controller import *