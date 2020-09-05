import os
import requests

from flask import jsonify, request

from run import app, model


@app.route('/')
@app.route('/index')
def index():
    return jsonify({
        "text": "Here is BERTServer!"
    })


@app.route('/predict', methods=['POST'])
def predict():
    response = {
        'success': False,
        'Content-Type': 'application/json'
    }
    if request.method == 'POST':
        if texts := request.get_json().get('texts'):
            if isinstance(texts, str):
                texts = [texts]
            elif isinstance(texts, list) and isinstance(texts[0], str):
                pass
            else:
                print('Key texts should be str or List[str].')
                return jsonify(response)

            outputs = model.extract(texts)
            response['prediction'] = outputs
            response['success'] = True

    return jsonify(response)
