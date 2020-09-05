import os
import requests

from flask import jsonify, request, render_template

from run import app, ncode_response_maker, text_response_maker


@app.route('/')
@app.route('/index')
def index():
    return jsonify('This is application!')


@app.route('/search_by_ncode', methods=['POST'])
def search_by_ncode():
    response = ncode_response_maker.make_response(request)
    return jsonify(response)


@app.route('/search_by_text', methods=['POST'])
def search_by_text():
    response = text_response_maker.make_response(request)
    return jsonify(response)