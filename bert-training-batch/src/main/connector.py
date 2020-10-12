from typing import Tuple

import MySQLdb
from MySQLdb.connections import Connection
from MySQLdb.cursors import Cursor
import pandas as pd
from pandas.core.frame import DataFrame

from config import Config


DB_BATCH_SIZE = Config.DB_BATCH_SIZE
DB_HOST_NAME = Config.DB_HOST_NAME
DB_PORT = Config.DB_PORT 


class DBConnector(object):
    """DBとの接続を担うクラス"""
    chunksize = DB_BATCH_SIZE
    host_name = DB_HOST_NAME
    port = DB_PORT

    @classmethod
    def get_conn_and_cursor(cls) -> Tuple[Connection, Cursor]:
        """DBへ接続するメソッド"""
        conn = MySQLdb.connect(
            host = cls.host_name,
            port = cls.port,
            user = 'root',
            password = 'root',
            database = 'maindb',
            use_unicode=True,
            charset="utf8"
        )
        cursor = conn.cursor()
        return conn, cursor

    
    @classmethod
    def get_db_df(cls, conn: Connection, latest_datetime: int, is_train: bool) -> DataFrame:
        is_train = int(is_train)
        query = f"SELECT ncode, title, text, genre, general_lastup \
                    FROM details \
                    WHERE general_lastup > {latest_datetime} AND bert_train = {is_train}"
        db_df_iterator = pd.read_sql_query(query, conn, chunksize=cls.chunksize)
        if db_df_list := list(db_df_iterator):
            db_df = pd.concat(db_df_list)
        else:
            db_df = pd.DataFrame()
        return db_df