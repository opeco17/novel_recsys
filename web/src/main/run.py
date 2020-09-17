import json
import logging
import logging.handlers

from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_mail import Mail

from config import Config


app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

handler = logging.handlers.RotatingFileHandler(app.config['LOG_FILE'], "a+", maxBytes=3000, backupCount=5)
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')) 
app.logger.addHandler(handler)
app.logger.setLevel(app.config['LOG_LEVEL'])

mail = Mail()
mail.init_app(app)

from controller import *