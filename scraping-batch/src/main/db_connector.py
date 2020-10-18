from typing import Tuple

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
        except Exception as e:
            extra = {'Class': 'DBConnector', 'Method': 'get_conn_and_cursor', 'Error': str(e)}
            logger.error('Unable to get DB connector and cursor.')
            raise
        return conn, cursor
    
    @classmethod
    def insert_details(cls, conn: Connection, cursor: Cursor, details_df: DataFrame):
        """作品の詳細情報をDBへ追加"""
        aligned_details_df = cls.__align_details_df(cursor, details_df)
        insert_data = [tuple(aligned_details_df.iloc[i]) for i in range(len(aligned_details_df))]
        cursor.executemany("INSERT IGNORE INTO details VALUES ({})".format(("%s, "*len(aligned_details_df.columns))[:-2]), insert_data)
        conn.commit()
        logger.info(f"{len(insert_data)} data was inserted to DB.")
    
    @classmethod
    def __align_details_df(cls, cursor: Cursor, details_df: DataFrame) -> DataFrame:
        """DBに投入するDataFrameをの順番をDBのカラムの順番と合わせる"""
        cursor.execute("SHOW columns FROM details")
        details_column_names = [column_name[0] for column_name in cursor.fetchall()]
        aligned_details_df = details_df[details_column_names]
        return aligned_details_df