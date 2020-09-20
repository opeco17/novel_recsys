import json
import os
import requests

from flask import request, Response

from models.bert import FeatureExtractor
from train.trainer import Trainer
from run import app


model = FeatureExtractor()


@app.route('/')
@app.route('/index')
def index():
    app.logger.info('index is called.')
    response_body = {"text": "Here is BERTServer!"}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    return response


@app.route('/predict', methods=['GET'])
def predict():
    app.logger.info('predict is called.')
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


@app.route('/train', methods=['GET'])
def train():
    app.logger.info('train is called.')
    success = Trainer.train()
    if success:
        status_code = 200
        response_body = {"message": "Training completed!"}
    else:
        status_code = 500
        response_body = {"message": "Training Failed."}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= status_code
    )
    return response