import json
import os

from flask import Flask

from config import Config
from scraping import TextScraper


app = Flask(__name__)
app.config.from_object(Config)

text_scraper = TextScraper(test=False)

from routes import *