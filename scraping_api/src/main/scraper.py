import argparse
import datetime
import gzip
import json
import time
from typing import List, Tuple, Any
from urllib.request import urlopen

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

from scraper_utils import *
from config import Config


NAROU_API_URL = Config.NAROU_API_URL
ELASTICSEARCH_HOST_NAME = Config.ELASTICSEARCH_HOST_NAME
FEATURE_EXTRACTION_URL = Config.FEATURE_EXTRACTION_URL
POINT_PREDICTION_URL = Config.POINT_PREDICTION_URL
DB_HOST_NAME = Config.DB_HOST_NAME


class DetailsScraper(object):
    """作品の詳細情報をスクレイピングするためのクラス
    
    Methods:
        scraping_details: 作品の詳細情報をスクレイピングするgenerator。
                                       mode='first'では最古の作品から最新の作品が対象。
                                       mode='middle'ではDB内で最新の作品から最新の作品が対象。
        __details_preprocessing: 作品の詳細情報を格納したDataFrameに対して前処理を行う。
        __get_scraping_item_number: スクレイピングの対象となる作品数を計算する。
    """
    
    def __init__(self, narou_api_url: str, batch_size: int=64, interval: float= 0.1, mode: str='first'):
        self.narou_api_url = narou_api_url
        self.batch_size = batch_size
        self.interval = interval
        self.mode = mode
        self.now = str(int(datetime.datetime.now().timestamp()))
        
    def scraping_details(self, cursor: Any=None) -> pd.core.frame.DataFrame:

        if self.mode == 'first':
            latest_registered_datetime = '1073779200'
        elif self.mode == 'middle' and cursor:
            cursor.execute("SELECT general_lastup FROM details ORDER BY general_lastup DESC LIMIT 1")
            sql_result = cursor.fetchone()
            latest_registered_datetime = str(sql_result[0]) if sql_result is not None else '1073779200'
        else:
            raise Exception('Argument mode should be middle or first.')

        lastup = self.now
        allcount = self.__get_scraping_item_number(latest_registered_datetime)
        all_queue_cnt = (allcount // self.batch_size)

        for i in range(all_queue_cnt):
            payload = {
                'out': 'json', 
                'gzip': 5, 
                'opt': 'weekly',
                'lim': self.batch_size,
                'lastup': latest_registered_datetime+"-"+str(lastup)
            }

            c = 0 # Avoid infinite loop
            while c < 5:
                try:
                    res = requests.get(self.narou_api_url, params=payload, timeout=30).content
                    break
                except:
                    print('Connection Error')
                    c += 1       

            r = gzip.decompress(res).decode('utf-8')

            details_df = pd.read_json(r).drop(0)
            details_df = self.__details_preprocessing(details_df)

            lastup = details_df.iloc[-1]["general_lastup"]

            time.sleep(self.interval)
            yield details_df

    def __details_preprocessing(self, details_df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        details_df = details_df.drop(['allcount', 'gensaku'], axis=1, errors='ignore')
        details_df = details_df.dropna(how='all')

        date_to_timestamp = lambda date: int(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp())

        for column in details_df.columns:
            if column in ['title', 'ncode', 'userid', 'writer', 'story', 'keyword']:
                details_df[column] = details_df[column].astype(str)
            elif column in['general_firstup', 'general_lastup', 'novelupdated_at', 'updated_at']:
                details_df[column] = details_df[column].map(str).map(date_to_timestamp)
            else:
                details_df[column] = details_df[column].astype(int)

        details_df['predict_point'] = 'Nan'
        details_df['text'] = 'Nan'

        return details_df

    def __get_scraping_item_number(self, latest_registered_datetime: str) -> int:
        payload = {
            'out': 'json', 
            'gzip': 5, 
            'of': 'n',
            'lim': 1, 
            'lastup': latest_registered_datetime+"-"+self.now
        }
        res = requests.get(self.narou_api_url, params=payload).content
        r =  gzip.decompress(res).decode("utf-8") 
        allcount = json.loads(r)[0]['allcount']
        return allcount


class TextScraper(object):
    """作品の本文をスクレイピングするためのクラス
    
    Methods:
        scraping_texts: 作品の本文をスクレイピングする。短編か連載かによって処理が異なる。
        __make_bs_obj: 引数のURLのBeautifulSoupオブジェクトを作成する。
        __get_main_text: BeautifulSoupオブジェクトを元に作品の本文をタグによって抽出する。
    """
    
    def __init__(self, narou_api_url: str, interval :float=0.1):
        self.narou_api_url = narou_api_url
        self.interval = interval
        
    def scraping_texts(self, ncodes: List[str]) -> Tuple[List[str]]:
        texts = []
        processed_ncodes = []

        for ncode in ncodes:
            time.sleep(self.interval)
            url = 'https://ncode.syosetu.com/' + ncode + '/'
            
            c = 0 # Avoid infinite loop
            while c < 5:
                try:
                    bs_obj = self.__make_bs_obj(url)
                    if bs_obj.findAll("dl", {"class": "novel_sublist2"}): # 連載作品の場合
                        url = 'https://ncode.syosetu.com/' + ncode + '/1/'
                        bs_obj = self.__make_bs_obj(url)
                    text = self.__get_main_text(bs_obj)  
                    break
                except Exception as e:
                    print(e)
                    c += 1
        
            processed_ncodes.append(ncode)
            texts.append(text)   
        return processed_ncodes, texts

    def __make_bs_obj(self, url: str) -> str:
        html = urlopen(url)
        return BeautifulSoup(html,"html.parser")

    def __get_main_text(self, bs_obj: BeautifulSoup) -> str:
        text = ""
        text_htmls = bs_obj.findAll('div', {'id': 'novel_honbun'})[0].findAll("p")

        for text_html in text_htmls:
            text = text + text_html.get_text() + "\n"

        return text


class Scraper(object):
    """スクレイピングとデータの投入を行うクラス
    
    Methods:
        scraping_and_add: 作品の詳細と本文をスクレイピングしデータの投入を行う。
        __add_to_database: 作品の詳細情報をDatabaseへ投入する。
        __add_to_elasticsearch: 作品の本文から抽出された特徴量をElasticsearchへ投入する。
    """
    
    def __init__(self, batch_size_of_scraper: int=64, batch_size_of_es: int=16, test=True):
        self.batch_size_of_scraper = batch_size_of_scraper
        self.batch_size_of_es = batch_size_of_es
        self.test = test
        
        self.conn, self.cursor = get_connector_and_cursor(DB_HOST_NAME)
        self.client = Elasticsearch(ELASTICSEARCH_HOST_NAME)
        self.details_scraper = DetailsScraper(NAROU_API_URL, batch_size_of_scraper, mode='first')
        self.text_scraper = TextScraper(NAROU_API_URL)

        self.data_count = 0
        
    def scraping_and_add(self):
        scraping_details_iterator = self.details_scraper.scraping_details()
        for i, details_df in enumerate(scraping_details_iterator):
            print((i + 1) * self.batch_size_of_scraper)
            ncodes, texts = self.text_scraper.scraping_texts(details_df.ncode)
            for ncode, text in zip(ncodes, texts):
                details_df.loc[details_df['ncode'] == ncode, 'text'] = text[:1024]
            predict_point = point_prediction(POINT_PREDICTION_URL, details_df)
            details_df['predict_point'] = predict_point
        
            self.__add_to_database(details_df)
            self.__add_to_elasticsearch(details_df)
            
            if self.test:
                break
        
        self.conn.close()
        self.client.close()

    def add_existing_data(self, chunksize=32):
        if self.test:
            details_df_iterator = pd.read_sql_query("SELECT * FROM details LIMIT 64", self.conn, chunksize=chunksize)
        else:
            details_df_iterator = pd.read_sql_query("SELECT * FROM details WHERE predict_point='Nan'", self.conn, chunksize=chunksize)
        for details_df in details_df_iterator:
            predict_point = point_prediction(POINT_PREDICTION_URL, details_df)
            details_df['predict_point'] = predict_point
            
            self.__update_database(details_df.ncode, details_df.predict_point)
            self.__add_to_elasticsearch(details_df)
            
            if self.test:
                break
        
        self.conn.close()
        self.client.close()
        
    def __add_to_database(self, details_df):
        self.cursor.execute("SHOW columns FROM details")
        columns_of_details = [column[0] for column in self.cursor.fetchall()]
        details_df_tmp = details_df[columns_of_details]
        details_data = [tuple(details_df_tmp.iloc[i]) for i in range(len(details_df_tmp))]
        self.cursor.executemany("INSERT IGNORE INTO details VALUES ({})".format(("%s, "*len(columns_of_details))[:-2]), details_data)
        self.conn.commit()

        self.data_count += len(details_data)
        
    def __update_database(self, ncodes, predict_points):
        data = [(ncode, predict_point) for ncode, predict_point in zip(ncodes, predict_points)]
        self.cursor.executemany("UPDATE details SET predict_point=%s WHERE ncode=%s", data)
        self.conn.commit()

        self.data_count += len(data)
        
    def __add_to_elasticsearch(self, details_df):
        recommendable_df = details_df[(details_df['predict_point'] == 1) & (details_df['global_point'] == 0)]
        if len(recommendable_df) != 0:
            for i in range(len(recommendable_df) // self.batch_size_of_es + 1):
                start, end = i * self.batch_size_of_es, (i + 1) * self.batch_size_of_es
                add_features_to_elasticsearch(
                    client=self.client, 
                    url=FEATURE_EXTRACTION_URL, 
                    df=recommendable_df.iloc[start:end],
                    h_dim=64,
                )
        print('{} data is inserted to Elasticsearch.'.format(len(recommendable_df)))
        
    def __del__(self):
        try:
            self.conn.close()
            self.client.close()
        except:
            pass
    
    
