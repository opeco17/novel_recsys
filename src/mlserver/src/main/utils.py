import json

from flask import Response

from logger import logger


def make_response(response_body: dict, status_code: int, message: str=None) -> Response:
    if message:
        response_body['message'] = message
        logger.info(message)
        
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status=status_code
    )
    return response