import os
import requests

from flask import jsonify, request, render_template

from run import app, similar_text_search


@app.route('/')
@app.route('/index')
def index():
    return jsonify('This is app.')


@app.route('/search_by_ncode', methods=['POST'])
def search_by_ncode():
    response = {
        'success': False,
        'Content-Type': 'application/json'
    }

    if request.method == 'POST':
        if request.get_json().get('ncode'):
            ncode = request.get_json().get('ncode')
            recommend_ncodes = similar_text_search.similar_search_by_ncode(ncode)
            response['recommend_ncodes'] = recommend_ncodes
            response['success'] = True

    return jsonify(response)


@app.route('/search_by_text', methods=['POST'])
def search_by_text():
    response = {
        'success': False,
        'Content-Type': 'application/json'
    }

    if request.method == 'POST':
        if request.get_json().get('text'):
            text = request.get_json().get('text')
            recommend_ncodes = similar_text_search.similar_search_by_text(text)
            response['recommend_ncodes'] = recommend_ncodes
            response['success'] = True

    return jsonify(response)