import json
import os
import requests

from flask import request

from run import app
from utils import make_response, ResponseMakerForNcodeAndText


ncode_response_maker = ResponseMakerForNcodeAndText('ncode')
text_response_maker = ResponseMakerForNcodeAndText('text')


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    app.logger.info('App: index called.')
    response_body = {'message': 'This is application!'}
    status_code = 200
    response = make_response(response_body, status_code)
    return response


@app.route('/search_by_ncode', methods=['GET'])
def search_by_ncode():
    app.logger.info('App: search_by_ncode called.')
    response_body, status_code, message = ncode_response_maker.make_response_body(request)
    response = make_response(response_body, status_code, message)
    return response


@app.route('/search_by_text', methods=['GET'])
def search_by_text():
    app.logger.info('App: search_by_text called.')
    response_body, status_code, message = text_response_maker.make_response_body(request)
    response = make_response(response_body, status_code, message)
    return response