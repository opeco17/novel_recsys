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
            response = session.post(url=Config.POP_DATA_URL, timeout=30)
        except requests.exceptions.ConnectTimeout:
            extra = {'Class': 'BatchManagerConnector', 'Method': 'get_queue_data'}
            logger.error('Unable to get queue data.')
        except Exception as e:
            extra = {'Class': 'BatchManagerConnector', 'Method': 'get_queue_data', 'Error': str(e)}
            logger.error('Unable to get queue data.')
            raise

        queue_data = response.json()['data']
        completions = response.json()['completions']
        return queue_data, completions
    
    @classmethod
    def fail_queue_data(cls, queue_data: int) -> None:
        cls.send_queue_data(queue_data, Config.FAIL_DATA_URL, 'fail')
        
    @classmethod
    def success_queue_data(cls, queue_data: int) -> None:
        cls.send_queue_data(queue_data, Config.SUCCESS_DATA_URL, 'success')
        
    @classmethod
    def send_queue_data(cls, queue_data: int, url: str, prefix: str) -> None:
        session = cls.__get_retry_session()
        headers = {'Content-Type': 'application/json'}
        data = {'data': queue_data}
        try:
            response = session.post(url=url, headers=headers, json=data, timeout=30)
        except requests.exceptions.ConnectTimeout:
            extra = {'Class': 'BatchManagerConnector', 'Method': f"{prefix}_queue_data"}
            logger.error(f"Unable to send {prefix} queue data.")
        except Exception as e:
            extra = {'Class': 'BatchManagerConnector', 'Method': f"{prefix}_queue_data", 'Error': str(e)}
            logger.error(f"Unable to send {prefix} queue data.")
            raise 
        
    @classmethod
    def __get_retry_session(cls) -> Session:
        """リクエストが失敗した時にリトライを行うためのセッションを提供"""
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session