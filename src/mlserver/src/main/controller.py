import json

from flask import request
import requests
import pandas as pd

from logger import logger
from run import app, feature_names, model
from utils import make_response


@app.route('/', methods=['GET'])
@app.route('/index', methods=['GET'])
def index():
    logger.info('index called.')
    response_body = {"message": "Here is MLServer!"}
    status_code = 200
    response = make_response(response_body, status_code)
    return response


@app.route('/predict', methods=['GET'])
def predict():
    """作品詳細の特徴量を元にポイント予測を行うAPI

    Fields:
        all_features_df: 作品詳細の全特徴量を保持したDataFrame
        features_df: ポイント予測に使用する特徴量を保持したDataFrame
        predicted_point: 作品のポイント予測結果 (0: 評価されない 1: 評価される)
    """
    logger.info('predict called.')
    response_body = {'success': False}
    status_code = 500
    
    if not (all_features:=request.get_json()):
        message = "Features is empty."
        response = make_response(response_body, status_code, message)
        return response
        
    if not isinstance(all_features, dict):
        all_features = json.loads(all_features)
    all_features_df = pd.DataFrame(all_features)
    
    for feature_name in feature_names:
        if feature_name not in list(all_features_df.columns):
            message = f"Lack of necessary feature: {feature_name}."
            response = make_response(response_body, status_code, message)
            return response

    features_df = all_features_df[feature_names]
    predicted_point = model.predict(features_df)
    response_body['prediction'] = predicted_point.tolist()
    response_body['success'] = True
    status_code = 200
    message = 'Prediction succeeded!'
    response = make_response(response_body, status_code, message)
    return response