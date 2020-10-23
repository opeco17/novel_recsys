import json
import sys

from flask import request, Response

from config import Config
from logger import logger
from queue_manager import QueueManager
from run import app


queue_manager = QueueManager(Config.COMPLETIONS)


@app.route('/', methods=['GET'])
def index():
    logger.info('Batch Manager: index called.')
    response_body = {"message": "Here is Batch Manager!"}
    status_code = 200
    response = Response(
        response=json.dumps(response_body),
        mimetype='application/json',
        status=200
    )
    return response


@app.route('/pop_data', methods=['POST'])
def pop_data():
    logger.info('Batch Manager: pop_data called.')
    data = queue_manager.pop_queue_data()
    response_body = {'data': data, 'completions': Config.COMPLETIONS}
    response = Response(
        response=json.dumps(response_body),
        mimetype='application/json',
        status=200
    )
    return response
    
    
@app.route('/fail_data', methods=['POST'])
def fail_data():
    logger.info('Batch Manager: fail_data called.')
    data = request.get_json().get('data')
    queue_manager.fail_queue_data(data)
    response = Response()
    return response


@app.route('/success_data', methods=['POST'])
def successdata():
    logger.info('Batch Manager: success_data called.')
    data = request.get_json().get('data')
    queue_manager.success_queue_data(data)
    response = Response()
    return response
