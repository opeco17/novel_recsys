from typing import Iterator, List, Tuple

import MySQLdb
from  MySQLdb.connections import Connection 
from MySQLdb.cursors import Cursor
import pandas as pd
from pandas.core.frame import DataFrame

from config import Config
from logger import logger


class DBConnector(object):
    """DBとの接続を担う"""
    
    @classmethod
    def get_conn_and_cursor(cls) -> Tuple[Connection, Cursor]:
        """DBのConnectorとCursorを提供"""
        try:
            conn = MySQLdb.connect(
                host = Config.DB_HOST_NAME,
                port = Config.DB_PORT,
                user = Config.DB_USER_NAME,
                password = Config.DB_PASSWORD,
                database = Config.DB_NAME,
                use_unicode=True,
                charset="utf8"
            )
            cursor = conn.cursor()
            logger.info('Get DB connector and cursor.')
            return conn, cursor
        
        except Exception as e:
            extra = {'Class': 'DBConnector', 'Method': 'get_conn_and_cursor', 'Error': str(e)}
            logger.error('Unable to get DB connector and cursor.')
            raise
        
    @classmethod
    def get_details_df_iterator(cls, conn: Connection, queue_data: int, completions: int, test: bool) -> Iterator[DataFrame]:
        """predicted_pointがNULLのミニバッチをイテレータで取得"""
        try:
            # test==Trueの際はpredicted_pointがNULL以外のレコードも対象とする。
            if test:
                query = f"SELECT title, story, text, keyword, \
                            novel_type, end, general_all_no, isr15, isbl, isgl, iszankoku, istensei, istenni, pc_or_k \
                            FROM details \
                            WHERE MOD(general_firstup, {completions})={queue_data}"
            else:
                query = f"SELECT title, story, text, keyword, \
                            novel_type, end, general_all_no, isr15, isbl, isgl, iszankoku, istensei, istenni, pc_or_k \
                            FROM details \
                            WHERE predicted_point is NULL AND MOD(general_firstup, {completions})={queue_data}"         
            
            details_df_iterator = pd.read_sql_query(
                sql=query,
                con=conn, 
                chunksize=Config.DETAILS_BATCH_SIZE
            )
            logger.info('Get details df iterator.')
            return details_df_iterator
        
        except Exception as e:
            extra = {'Class': 'DBConnector', 'Method': 'get_details_df_iterator', 'Error': str(e)}
            logger.error('Unable to get details df iterator.')
            raise
        
    @classmethod
    def update_predicted_point(cls, conn: Connection, cursor: Cursor, ncodes: List[str], predict_points: List[int]) -> None:
        """DB内の作品の予測ポイントを更新"""
        data = [(ncode, predict_point) for ncode, predict_point in zip(ncodes, predict_points)]
        cursor.executemany("UPDATE details SET predicted_point=%s WHERE ncode=%s", data)
        conn.commit()