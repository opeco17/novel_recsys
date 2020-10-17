from typing import Tuple
from urllib3.util import Retry

import requests
from requests.adapters import HTTPAdapter
from requests.sessions import Session

from config import Config

from logger import logger


class BatchManagerConnector(object):
    
    @classmethod
    def get_queue_data(cls) -> Tuple[int, int]:
        session = cls.__get_retry_session()
        try:
            response = session.post(url=Config.POP_DATA_URL, timeout=10)
        except requests.exceptions.ConnectTimeout:
            extra = {'Class': 'BatchManagerConnector', 'Method': 'get_queue_data'}
            logger.error('Unable to get scraping_item_number')
        except Exception as e:
            extra = {'Class': 'BatchManagerConnector', 'Method': 'get_queue_data', 'Error': e}
            logger.error('Unable to get queue data.')
            raise

        queue_data = response.json()['data']
        completions = response.json()['completions']
        return queue_data, completions
    
    @classmethod
    def delete_queue_data(cls, queue_data: int):
        session = cls.__get_retry_session()
        headers = {'Content-Type': 'application/json'}
        data = {'data': queue_data}
        try:
            response = session.post(url=Config.DELETE_DATA_URL, headers=headers, json=data, timeout=10)
        except requests.exceptions.ConnectTimeout:
            extra = {'Class': 'BatchManagerConnector', 'Method': 'delete_queue_data'}
            logger.error('Unable to get scraping_item_number')
        except Exception as e:
            extra = {'Class': 'BatchManagerConnector', 'Method': 'delete_queue_data', 'Error': e}
            logger.error('Unable to delete queue data.')
            raise 
        
    @classmethod
    def __get_retry_session(cls) -> Session:
        """リクエストが失敗した時にリトライを行うためのセッションを提供"""
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session