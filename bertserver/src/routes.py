import os
import requests

from flask import jsonify, request

from run import app, model


@app.route('/')
def index():
    return jsonify({
        "message": "Here is index!"
    })


@app.route('/predict', methods=['POST'])
def predict():
    response = {
        'success': False,
        'Content-Type': 'application/json'
    }

    if request.method == 'POST':
        if request.get_json().get('texts'):
            texts = request.get_json().get('texts')
            outputs = model.extract(texts)
            response['prediction'] = outputs
            response['success'] = True
    return jsonify(response)
