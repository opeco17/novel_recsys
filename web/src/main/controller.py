import json

from flask import request, Response

from run import app


@app.route('/')
@app.route('/index', methods=['GET'])
def index():
    response_body = {'text': 'This is Web!'}
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= 200
    )
    app.logger.info('Response body: ' + str(response_body))
    return response