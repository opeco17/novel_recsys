from typing import List

import requests

from config import Config
from logger import logger


class BERTServerConnector(object):
    """BERTServerとの接続を担うクラス"""

    @classmethod
    def extract_features(cls, texts: List[str]) -> List[float]:
        headers = {'Content-Type': 'application/json'}
        data = {'texts': texts}
        try:
            response = requests.get(Config.FEATURE_EXTRACTION_URL, headers=headers, json=data)
        except Exception as e:
            extra = {'Class': 'BERTServerConnector', 'Method': 'extract_features', 'ErrorType': type(e), 'Error': str(e)}
            logger.error('Unable to extract features.')
            raise
        features = response.json()['prediction']
        logger.info('Feature extraction succeeded.')
        return features