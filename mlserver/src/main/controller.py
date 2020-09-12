import json

from flask import request, Response
import requests
import pandas as pd

from run import app, feature_names, model


@app.route('/')
@app.route('/index')
def index():
    response_body = {"message": "Here is MLServer!"}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    return response


@app.route('/predict', methods=['GET'])
def predict():
    """作品詳細の特徴量を元にポイント予測を行うAPI

    Fields:
        all_features_df: 作品詳細の全特徴量を保持したDataFrame
        features_df: ポイント予測に使用する特徴量を保持したDataFrame
        predicted_point: 作品のポイント予測結果 (0: 評価されない 1: 評価される)
    """
    response_body = {"success": False}
    status_code = 500
    if all_features:=request.get_json():
        if not isinstance(all_features, dict):
            all_features = json.loads(all_features)
        all_features_df = pd.DataFrame(all_features)
        flag = True
        for feature_name in feature_names:
            if feature_name not in list(all_features_df.columns):
                flag = False
                response_body['message'] = 'Lack of necessary feature.'
                break
        if flag:
            features_df = all_features_df[feature_names]
            predicted_point = model.predict(features_df)
            response_body['prediction'] = predicted_point.tolist()
            response_body['success'] = True
            status_code = 200

    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= status_code
    )
    app.logger.info('Response body: ' + str(response_body))
    return response