import json
from typing import List

from pandas.core.frame import DataFrame
import requests

from config import Config
from logger import logger


class MLServerConnector(object):
    """MLServerとの接続を担うクラス"""

    @classmethod
    def predict_point(cls, details_df: DataFrame) -> List[int]:   
        """作品にポイントが付くか否かを二値予測"""     
        headers = {'Content-Type': 'application/json'}
        data = {column: list(details_df[column]) for column in list(details_df.columns)}
        data = json.dumps(data)
        response = requests.get(Config.POINT_PREDICTION_URL, headers=headers, json=data)
        predicted_points = response.json()['prediction']
        logger.info(f"{len(predicted_points)} data was predicted.")
        logger.info(f"Predicted point is {predicted_points}")
        return predicted_points