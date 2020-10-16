import json

import requests

from config import Config
from logger import logger


class Messenger(object):

    @classmethod
    def send_message(cls, text):    
        webhook_url = Config.WEBHOOK_URL
        if webhook_url:
            username = 'Scraping batch'
            data = json.dumps({'text': text, 'username': username})
            try:
                requests.post(webhook_url, data=data)
                logger.info('Send message succeeded.')
            except Exception as e:
                extra = {'Error': e}
                logger.info('Send message failed.', extra=extra)
        else:
            logger.info('No webhook url.')