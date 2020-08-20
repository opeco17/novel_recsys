import json
import requests

import pandas as pd

# 全データをまとめて送信すると落ちるのでサイズ64のミニバッチに分割


df = pd.read_csv('../../data_analysis/dataset/csv/detail_text_with_sup.csv')

url = 'http://localhost:5000/predict'
headers = {'Content-Type': 'application/json'}
data = {}
data = {column: list(df[column])[:5] for column in list(df.columns)}
data = json.dumps(data)
r_post = requests.post(url, headers=headers, json=data)

