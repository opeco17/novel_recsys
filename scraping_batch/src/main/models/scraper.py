import sys
from pandas.core.frame import DataFrame
from typing import List, Tuple, Any
sys.path.append('..')

from run import app
from config import Config
from models.connector import ElasticsearchConnector, DBConnector, MLServerConnector
from models.preprocesser import Preprocesser
from models.sub_scraper import DetailsScraper, TextScraper


class Scraper(object):
    """スクレイピングとデータの投入を行うクラス"""
    
    def __init__(self):
        self.conn, self.cursor = DBConnector.get_conn_and_cursor()
        self.client = ElasticsearchConnector.get_client()
        self.db_count = 0
        self.es_count = 0
        app.logger.info('Scraper constructed!')
        
    def scraping_and_add(self, test: bool, mode: str, epoch: int):
        """スクレイピングを行いポイント予測とDatabaseへの追加とElasticsearchへの追加を行う"""
        scraped_details_iterator = DetailsScraper.get_scraped_df_iterator(mode, self.conn, self.cursor)
        for i, sub_details_df in enumerate(scraped_details_iterator):
            ncodes, texts = TextScraper.scraping_texts(sub_details_df.ncode)
            for ncode, text in zip(ncodes, texts):
                sub_details_df.loc[sub_details_df['ncode']==ncode, 'text'] = text[:1024]
            sub_details_df = Preprocesser.preprocess_details(sub_details_df)
            sub_details_df = Preprocesser.preprocess_ml_details(sub_details_df)
            predicted_point = MLServerConnector.predict_point(sub_details_df)
            sub_details_df['predict_point'] = predicted_point

            self.__add_to_database(sub_details_df)
            self.__add_to_elasticsearch(sub_details_df)

            if test and i + 1 == epoch:
                break
        
        app.logger.info('[scraping and add] processes are completed!')
        self.conn.close()
        self.client.close()

    def add_existing_data(self, test: bool, epoch: int):
        """Database内の既存データに対してポイント予測とElasticsearchへの追加を行う"""
        details_df_iterator = DBConnector.get_details_df_iterator(self.conn, test, epoch)
        for i, sub_details_df in enumerate(details_df_iterator):
            app.logger.info(sub_details_df)
            sub_details_df = Preprocesser.preprocess_ml_details(sub_details_df)
            predicted_point = MLServerConnector.predict_point(sub_details_df)
            sub_details_df['predict_point'] = predicted_point

            self.__update_database(sub_details_df.ncode, sub_details_df['predict_point'])
            self.__add_to_elasticsearch(sub_details_df)

            if test and i + 1 == epoch:
                break
            
        app.logger.info('[add existing data] processes are completed!')
        self.conn.close()
        self.client.close()
        
    def __add_to_database(self, details_df: DataFrame):
        """作品の詳細情報全てをDatabaseへ追加"""
        DBConnector.add_details(self.conn, self.cursor, details_df)
        self.db_count += len(details_df)
        app.logger.info(f'{len(details_df)} data was inserted to database.')
        
    def __update_database(self, ncodes: List[str], predict_points: List[int]):
        """Databaseの予測ポイントを更新"""
        DBConnector.update_predict_points(self.conn, self.cursor, ncodes, predict_points)
        self.db_count += len(ncodes)
        app.logger.info(f'{len(ncodes)} data was updated in database.')
        
    def __add_to_elasticsearch(self, details_df: DataFrame):
        """小説の詳細情報の一部と特徴量をElasticsearchへ追加(indexが存在しない場合は新規作成)"""
        if not self.client.indices.exists(index='features'):
            ElasticsearchConnector.create_indices(self.client)
        ElasticsearchConnector.add_details(self.client, details_df)
        self.es_count += len(details_df)
        app.logger.info(f'{len(details_df)} data was inserted to Elasticsearch.')
        
    def __del__(self):
        """インスタンスが破棄される際にDBとの接続を切る"""
        try:
            self.conn.close()
            self.client.close()
        except:
            pass



    
    
