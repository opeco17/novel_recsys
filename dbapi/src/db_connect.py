import re
import sqlite3


def create_connection(db_path, extension_path):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(extension_path)
    c = conn.cursor()
    return conn, c
 

class DBConnector(object):

    # connect to database
    def __init__(self, db_path, extension_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.enable_load_extension(True)
        self.conn.load_extension(extension_path)
        self.cur = self.conn.cursor()

    # access to features table
    def initialize_features(self):
        self.cur.execute("DROP TABLE IF EXISTS features")
        self.cur.execute("CREATE TABLE features (ncode VARCHAR, feature_index INTEGER, feature FLOAT32)")        

    def get_ncodes_from_features(self):
        self.cur.execute("SELECT DISTINCT ncode FROM features")
        ncodes = [ncode[0] for ncode in self.cur.fetchall()]
        return ncodes

    def insert_features(self, ncode, features):
        features_data = [(ncode, feature_index, float(feature)) for feature_index, feature in enumerate(features)]
        self.cur.executemany("INSERT INTO features VALUES (?, ?, ?)", features_data)

    # access to details table
    def initialize_details_by_dataframe(self, df):
        self.cur.execute("DROP TABLE IF EXISTS details")
        df.to_sql('details', self.conn)

    def insert_details_from_dataframe(self, df):
        df.to_sql('details', self.conn, if_exists='append')

    def get_registered_latest_datetime(self):
        self.cur.execute('SELECT general_lastup FROM details ORDER BY general_lastup DESC LIMIT 1')
        return self.cur.fetchone()[0]

    def get_all_ncodes_from_details(self):
        self.cur.execute("SELECT DISTINCT ncode FROM details")
        ncodes = [ncode[0] for ncode in self.cur.fetchall()]
        return ncodes

    def get_ncodes_of_null_text(self):
        self.cur.execute("SELECT ncode FROM details WHERE text='Nan'")
        ncodes = [ncode[0] for ncode in self.cur.fetchall()]
        return ncodes

    def update_texts(self, ncodes, texts):
        texts = list(map(lambda text: re.sub("\'", "\"", text), texts))
        for ncode, text in zip(ncodes, texts):
            text = re.sub("\'", "\"", text)
            self.cur.execute("UPDATE details SET text='{}' WHERE ncode='{}' AND text='Nan'".format(text, ncode))

    def insert_details(self, ncodes, stories, all_points, predict_points):
        details_data = [(ncode, story, int(all_point), int(predict_point)) for ncode, story, all_point, predict_point in zip(ncodes, stories, all_points, predict_points)]
        self.cur.executemany("INSERT INTO details VALUES (?, ?, ?, ?)", details_data)

    # close database connection
    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()


