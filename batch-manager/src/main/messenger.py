import json

import requests

from config import Config
from run import app


class Messenger(object):

    @classmethod
    def send_message(cls, username: str, text: str) -> None:    
        webhook_url = Config.WEBHOOK_URL
        if webhook_url:
            data = json.dumps({'text': text, 'username': username})
            try:
                requests.post(webhook_url, data=data)
                app.logger.info('Send message succeeded.')
            except Exception as e:
                extra = {'Error': e}
                app.logger.info('Send message failed.', extra=extra)
        else:
            app.logger.info('No webhook url.')