import os
import requests

from flask import request

from logger import logger
from models.bert import FeatureExtractor
from run import app
from utils import make_response


model = FeatureExtractor()


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    logger.info('index called.')
    response_body = {"message": "Here is BERTServer!"}
    status_code = 200
    response = make_response(response_body, status_code)
    return response


@app.route('/predict', methods=['GET'])
def predict():
    logger.info('predict called.')
    status_code = 500
    response_body = {"success": False}
    
    if not (texts := request.get_json().get('texts')):
        response = make_response(response_body, status_code)
        return response
    
    if not ((isinstance(texts, str)) or (isinstance(texts, list) and isinstance(texts[0], str))):
        status_code = 500
        logger.info('Key texts should be str or List[str].')
        response = make_response(response_body, status_code)
        return response

    texts = [texts] if isinstance(texts, str) else texts
    outputs = model.extract(texts)
    response_body['prediction'] = outputs
    response_body['success'] = True
    status_code = 200
    logger.info('Prediction succeeded!')
    response = make_response(response_body, status_code)
    return response