import logging

import requests

# Get Request
url = 'http://mlserver:3032'
r_get = requests.get(url)
contents = r_get.json()
print(contents)

# Post Request
url = 'http://mlserver:3032/predict'
headers = {'Content-Type': 'application/json'}
data = {'text': 'I am a BERT.'}
r_post = requests.post(url, headers=headers, json=data)
print(r_post)
contents = r_post.json()
print(contents)
