import MySQLdb

def get_connector_and_cursor():
    conn = MySQLdb.connect(
        host = 'database',
        port = 3306,
        user = 'root',
        password = 'root',
        database = 'maindb'
    )
    cursor = conn.cursor()
    return conn, cursor


get_connector_and_cursor()
