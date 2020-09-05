import os
import requests

from flask import jsonify, request

from run import app, text_scraper


@app.route('/')
@app.route('/index')
def index():
    return jsonify({
        'message': 'Here is ScrapingApi!'
    })


@app.route('/scraping_texts', methods=['POST'])
def scraping_texts():
    response = {
        'success': False,
        'Content-Type': 'application/json'
    }

    if request.method == 'POST':
        if request.get_json().get('ncodes'):
            ncodes = request.get_json().get('ncodes')
            processed_ncodes, texts = text_scraper.scraping_texts(ncodes)
            response['ncodes'] = processed_ncodes
            response['texts'] = texts
            response['success'] = True
    return jsonify(response)