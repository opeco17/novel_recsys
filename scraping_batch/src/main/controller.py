import json
import os
import requests

from flask import jsonify, request, Response

from run import app
from models.scraper import Scraper


@app.route('/')
def index():
    response_body = {"message": "ScrapingApi!"}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    return response


@app.route('/scraping_and_add', methods=['POST'])
def scraping_and_add():
    """スクレイピングを行い、得られたデータを処理してDBとElasticsearchへ登録するバッチ処理"""
    response_body = {"success": False}
    status_code = 500

    if (host := os.environ.get('HOST')) == None:
        host = 'local'
    
    if (test:=request.get_json().get('test')) and (mode:=request.get_json().get('mode')):
        if isinstance(test, bool) and mode in ['first', 'middle']:
            scraper = Scraper(test=test)
            scraper.scraping_and_add(mode=mode)
            response_body['success'] = True
            response_body["message"] = f"{scraper.data_count} data were scraped and registered."
            status_code = 200
        else:
            response_body["message"] = "test should be boolean and mode should be str."
    else:
        response_body["message"] = "Please specify test(true or false) and mode(first or middle)."

    response = Response(
        response=json.dumps(response_body),
        mimetype='application/json',
        status=status_code
    )
    return response
        

@app.route('/add_existing_data', methods=['POST'])
def add_existing_data():
    """既存のデータを処理してDBとElasticsearchへ登録するバッチ処理"""
    response_body = {"success": False}
    status_code = 500

    if (host := os.environ.get('HOST')) == None:
        host = 'local'
    
    if test := request.get_json().get('test'):
        if isinstance(test, bool):
            scraper = Scraper(test=test)
            scraper.add_existing_data()
            response_body['success'] = True
            response_body["message"] = f"{scraper.data_count} data were registered."
            status_code = 200
        else:
            response_body["message"] = "test should be boolean."
    else:
        response_body["message"] = "Please specify test using boolean."

    response = Response(
        response=json.dumps(response_body),
        mimetype='application/json',
        status=status_code
    )
    return response


