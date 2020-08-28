import os
import requests

from flask import request, jsonify

from run import app
from scraping import scraping_texts


@app.route('/')
def index():
    return jsonify({
        "message": "Here is scraping api!"
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
            processed_ncodes, texts = scraping_texts(ncodes, test=False)
            response['ncodes'] = processed_ncodes
            response['texts'] = texts
            response['success'] = True
    return jsonify(response)


@app.route('/scraping_details', methods=['POST'])
def scraping_details():
    response = {
        'success': False,
        'Content-Type': 'application/json'
    }

    if request.method == 'POST':
        if request.get_json().get('ncodes'):
            ncodes = request.get_json().get('ncodes')
            processed_ncodes, texts = scraping_texts(ncodes, test=False)
            response['ncodes'] = processed_ncodes
            response['texts'] = texts
            response['success'] = True
    return jsonify(response)