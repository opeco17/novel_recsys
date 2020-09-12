import json
import os
import requests

from flask import request, Response

from models.bert import FeatureExtractor
from run import app


model = FeatureExtractor()


@app.route('/')
@app.route('/index')
def index():
    response_body = {"text": "Here is BERTServer!"}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    return response


@app.route('/predict', methods=['GET'])
def predict():
    response_body = {"success": False,}
    status_code = 500
    if texts := request.get_json().get('texts'):
        if (flag:=isinstance(texts, str)) or (isinstance(texts, list) and isinstance(texts[0], str)):
            texts = [texts] if flag else texts
            outputs = model.extract(texts)
            response_body['prediction'] = outputs
            response_body['success'] = True
            status_code = 200
        else:
            app.logger.info('Key texts should be str or List[str].')
            status_code = 500

    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= status_code
    )
    app.logger.info(f'Response body: {response_body}')
    return response
