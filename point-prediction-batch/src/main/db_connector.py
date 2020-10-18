from typing import Tuple

import MySQLdb
from  MySQLdb.connections import Connection 
from MySQLdb.cursors import Cursor
import pandas as pd
from pandas.core.frame import DataFrame

from config import Config
from logger import logger


DB_NAME = Config.DB_NAME
DB_USER_NAME = Config.DB_USER_NAME
DB_PASSWORD = Config.DB_PASSWORD
DB_HOST_NAME = Config.DB_HOST_NAME
DB_PORT = Config.DB_PORT


class DBConnector(object):
    """DBとの接続を担う"""
    
    @classmethod
    def get_conn_and_cursor(cls) -> Tuple[Connection, Cursor]:
        """DBのConnectorとCursorを提供"""
        try:
            conn = MySQLdb.connect(
                host = DB_HOST_NAME,
                port = DB_PORT,
                user = DB_USER_NAME,
                password = DB_PASSWORD,
                database = DB_NAME,
                use_unicode=True,
                charset="utf8"
            )
            cursor = conn.cursor()
            logger.info('Get DB connector and cursor.')
        except Exception as e:
            extra = {'Class': 'DBConnector', 'Method': 'get_conn_and_cursor', 'Error': e}
            logger.error('Unable to get DB connector and cursor.')
            raise
        return conn, cursor
    
    @classmethod
    def update_predict_points(cls, conn: Connection, cursor: Cursor, ncodes: List[str], predict_points: List[int]) -> None:
        """作品の予測ポイントを更新"""
        data = [(ncode, predict_point) for ncode, predict_point in zip(ncodes, predict_points)]
        cursor.executemany("UPDATE details SET predict_point=%s WHERE ncode=%s", data)
        conn.commit()