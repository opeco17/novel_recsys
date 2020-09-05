import json
import os

from flask import jsonify, request
import requests
import pandas as pd

from run import app, feature_names, model


@app.route('/')
@app.route('/index')
def index():
    return jsonify({"message": "Here is MLServer!"})


@app.route('/predict', methods=['POST'])
def predict():
    """作品詳細の特徴量を元にポイント予測を行うAPI

    Fields:
        all_features_df: 作品詳細の全特徴量を保持したDataFrame
        features_df: ポイント予測に使用する特徴量を保持したDataFrame
        predicted_point: 作品のポイント予測結果 (0: 評価されない 1: 評価される)
    """
    response = {
        'success': False,
        'Content-Type': 'application/json'
    }
    if request.method == 'POST':
        if all_features:=request.get_json():
            if not isinstance(all_features, dict):
                all_features = json.loads(all_features)
            all_features_df = pd.DataFrame(all_features)
            for feature_name in feature_names:
                if feature_name not in list(all_features_df.columns):
                    response['message'] = 'Lack of necessary feature.'
                    return jsonify(response)
            features_df = all_features_df[feature_names]
            predicted_point = model.predict(features_df)
            response['prediction'] = predicted_point.tolist()
            response['success'] = True
    return jsonify(response)