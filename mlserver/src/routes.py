import json
import os

from flask import jsonify, request
import requests
import pandas as pd

from run import app, feature_names, model


@app.route('/')
def index():
    return jsonify({
        "message": "Here is index!"
    })


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
        if request.get_json():
            all_features = request.get_json()
            all_features_df = pd.DataFrame(json.loads(all_features))
            features_df = all_features_df[feature_names]
            predicted_point = model.predict(features_df)
            response['prediction'] = predicted_point.tolist()
            response['success'] = True
    return jsonify(response)