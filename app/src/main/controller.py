import json
import os
import requests

from flask import request, Response

from run import app
from utils import ResponseMakerForNcodeAndText

ncode_response_maker = ResponseMakerForNcodeAndText('ncode')
text_response_maker = ResponseMakerForNcodeAndText('text')

@app.route('/')
@app.route('/index', methods=['GET'])
def index():
    response_body = {'text': 'This is application!'}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    app.logger.info('Response body: ' + str(response_body))
    return response


@app.route('/search_by_ncode', methods=['GET'])
def search_by_ncode():
    response_body, status_code = ncode_response_maker.make_response_body(request)
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= status_code
    )
    app.logger.info('Response body: ' + str(response_body))
    return response


@app.route('/search_by_text', methods=['GET'])
def search_by_text():
    response_body, status_code = text_response_maker.make_response_body(request)
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= status_code
    )
    app.logger.info('Response body: ' + str(response_body))
    return response