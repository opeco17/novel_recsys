import json

from flask import Response


def make_response(response_body: dict, status_code: int) -> Response:
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status= status_code
    )
    return response