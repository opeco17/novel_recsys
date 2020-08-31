import os
import requests

from flask import jsonify, request, render_template

from run import app, similar_item_search


@app.route('/')
@app.route('/index')
def index():
    return jsonify('This is application!')


@app.route('/search_by_ncode', methods=['POST'])
def search_by_ncode():

    response = {
        'success': False,
        'recommend_ncodes': [],
        'Content-Type': 'application/json',
    }

    if request.method == 'POST':
        if request.get_json():
            if ncode := request.get_json().get('ncode'):
                if (ncode_type := type(ncode)) is str:
                    recommend_list = similar_item_search.similar_search_by_ncode(ncode)
                    response['recommend_list'] = recommend_list
                    response['success'] = True
                else:
                    response['message'] = f'Parameter ncode should be str but you throw {ncode_type}.'
            else:
                keys = list(request.get_json().keys())
                response['message'] = f'Parameter should be ncode but you throw {keys}.'
        else:
            response['message'] = 'Parameter should be ncode but you throw none.'

    return jsonify(response)


@app.route('/search_by_text', methods=['POST'])
def search_by_text():

    response = {
        'success': False,
        'recommend_ncodes': [],
        'Content-Type': 'application/json',
    }

    if request.method == 'POST':
        if request.get_json():
            if text := request.get_json().get('text'):
                if (text_type := type(text)) is str:
                    text = [text]
                    recommend_list = similar_item_search.similar_search_by_text(text)
                    response['recommend_list'] = recommend_list
                    response['success'] = True
                else:
                    response['message'] = f'Parameter text should be str but you throw {ncode_type}.'    
            else:
                keys = list(request.get_json().keys())
                response['message'] = f'Parameter should be text but you throw {keys}.'
        else:
            response['message'] = 'Parameter should be text but you throw none.'

    return jsonify(response)