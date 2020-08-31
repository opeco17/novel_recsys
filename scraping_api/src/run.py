import json
import os

from flask import Flask

from config import Config
from scraper import TextScraper


NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'

app = Flask(__name__)
app.config.from_object(Config)

text_scraper = TextScraper(NAROU_API_URL)

from routes import *