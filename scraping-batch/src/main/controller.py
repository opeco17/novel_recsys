import os
import requests

from flask import request

from run import app
from models.scraper import Scraper
from utils import make_response


@app.route('/')
def index():
    app.logger.info('Scraping Batch: index called.')
    response_body = {'message': 'ScrapingApi!'}
    status_code = 200
    response = make_response(response_body, status_code)
    return response


@app.route('/scraping_and_add', methods=['POST'])
def scraping_and_add():
    """スクレイピングを行い、得られたデータを処理してDBとElasticsearchへ登録するバッチ処理"""
    app.logger.info('Scraping Batch: scraping_and_add called.')
    response_body = {"success": False}
    status_code = 500
    
    host = os.environ.get('HOST', 'local')
    
    if (test:=request.get_json().get('test')) == None:
        message = "Please specify test as bool."
        response = make_response(response_body, status_code, message)
        return response
        
    if not (mode:=request.get_json().get('mode')):
        message = "Please specify mode as str (first or middle)."
        response = make_response(response_body, status_code, message)
        return response

    if not isinstance(test, bool):
        message = f"Parameter test should be bool but you throw {type(test)}."
        response = make_response(response_body, status_code, message)
        return response

    if not (mode in ['first', 'middle']):
        message = f"Parameter mode should be first or middle but you throw {mode}."
        response = make_response(response_body, status_code, message)
        return response

    if not (test == True and isinstance(epoch:=request.get_json().get('epoch'), int) and epoch > 0):
        message = "When test is True, epoch should be a natural number."
        response = make_response(response_body, status_code, message)
        return response
    
    if test == False:
        epoch = None
        
    app.logger.info(f"Start scraping (test: {test}  mode: {mode}  epoch: {epoch})")
    scraper = Scraper()
    result = {}
    try:
        scraper.scraping_and_add(test=test, mode=mode, epoch=epoch, result=result)
        response_body['success'] = True
        message = f"{scraper.db_count} and {scraper.es_count} data were registeres to DB and ES respectively."
        status_code = 200
        response = make_response(response_body, status_code, message)
        return response
    
    except Exception as e:
        message = f"Failed: {e}."
        response = make_response(response_body, status_code, message)
        return response
    

@app.route('/add_existing_data', methods=['POST'])
def add_existing_data():
    """既存のデータを処理してDBとElasticsearchへ登録するバッチ処理"""
    app.logger.info('Scraping Batch: add_existing_data called.')
    response_body = {"success": False}
    status_code = 500

    host = os.environ.get('HOST', 'local')
    
    if (test:=request.get_json().get('test')) == None:
        message = "Please specify test as bool."
        response = make_response(response_body, status_code, message)
        return response
        
    if not isinstance(test, bool):
        message = f"Parameter test should be bool but you throw {type(test)}."
        response = make_response(response_body, status_code, message)
        return response
    
    if test == True and not (isinstance(epoch:=request.get_json().get('epoch'), int) and epoch > 0):
        message = "When test is True, epoch should be a natural number."
        response = make_response(response_body, status_code, message)
        return response
    
    if test == False:
        epoch = None
        
    app.logger.info(f"Start scraping (test: {test}  epoch: {epoch})")
    scraper = Scraper()
    result = {}
    try:
        scraper.add_existing_data(test=test, epoch=epoch, result=result)
        response_body['success'] = True
        message = f"{scraper.db_count} and {scraper.es_count} data were registeres to DB and ES respectively."
        status_code = 200
        response = make_response(response_body, status_code, message)
        return response
        
    except Exception as e:
        message = f"Failed: {e}."
        response = make_response(response_body, status_code, message)
        return response


