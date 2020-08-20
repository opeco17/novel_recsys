import os
import requests

from flask import request, render_template, jsonify

from run import app


@app.route('/')
@app.route('/index')
def index():
    url = 'http://mlserver:3032'
    r_get = requests.get(url)
    contents = r_get.json()
    return jsonify(contents)


@app.route('/predict')
def predict():
    url = 'http://mlserver:3032/predict'
    headers = {'Content-Type': 'application/json'}
    data = {'text': 'I am a BERT.'}
    r_post = requests.post(url, headers=headers, json=data)
    contents = r_post.json()
    return jsonify(contents)