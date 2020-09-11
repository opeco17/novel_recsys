import json
import os
import requests

from flask import jsonify, request, render_template, Response

from run import app, ncode_response_maker, text_response_maker


@app.route('/')
@app.route('/index', methods=['GET'])
def index():
    response_body = {'text': 'This is application!'}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    return response


@app.route('/search_by_ncode', methods=['GET'])
def search_by_ncode():
    response_body = ncode_response_maker.make_response(request)
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    return response


@app.route('/search_by_text', methods=['GET'])
def search_by_text():
    response_body = text_response_maker.make_response(request)
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    return response