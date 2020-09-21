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
    app.logger.info('BERTServer: index called.')
    response_body = {"message": "Here is BERTServer!"}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    return response


@app.route('/predict', methods=['GET'])
def predict():
    app.logger.info('BERTServer: predict called.')
    response_body = {"success": False,}
    if texts := request.get_json().get('texts'):
        if (flag:=isinstance(texts, str)) or (isinstance(texts, list) and isinstance(texts[0], str)):
            texts = [texts] if flag else texts
            outputs = model.extract(texts)
            response_body['prediction'] = outputs
            response_body['success'] = True
            status_code = 200
            app.logger.info('Prediction succeeded!')
        else:
            status_code = 500
            app.logger.info('Key texts should be str or List[str].')
            
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= status_code
    )
    return response


@app.route('/train', methods=['GET'])
def train():
    app.logger.info('BERTServer: train called.')
    success = Trainer.train()
    if success:
        status_code = 200
        message = "Training completed!"
        response_body = {"message": message}
        app.logger.info(message)
    else:
        status_code = 500
        message = "Training Failed."
        response_body = {"message": message}
        app.logger.info(message)

    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= status_code
    )
    return response