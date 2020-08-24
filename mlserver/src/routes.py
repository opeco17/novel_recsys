import os
import json
import requests

import pandas as pd
from flask import request, jsonify

from run import app, model, feature_names


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
        if request.get_json():
            '''
            json all_features: pair of columns and value of list
            pd.DataFrame fetures: features of narou novel
            np.ndarray predicted_point: point of narou novel predicted by machine learning
            '''
            all_features = request.get_json()
            all_features = pd.DataFrame(json.loads(all_features))
            features = all_features[feature_names]
            predicted_point = model.predict(features)
            response['prediction'] = predicted_point.tolist()
            response['success'] = True
    return jsonify(response)