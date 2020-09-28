import MySQLdb
from  MySQLdb.connections import Connection 
from MySQLdb.cursors import Cursor

def get_conn_and_cursor(cls) -> Tuple[Connection, Cursor]:
    """DBへ接続するメソッド"""
    conn = MySQLdb.connect(
        host = 'database',
        port = 3306,
        user = 'root',
        password = 'root',
        database = 'maindb',
        use_unicode=True,
        charset="utf8"
    )
    cursor = conn.cursor()
    return conn, cursor

conn, cursor = get_conn_and_cursor()