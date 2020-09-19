import datetime
import gzip
import json
import sys
import time
from typing import List, Tuple, Any
from urllib.request import urlopen
sys.path.append('..')

from bs4 import BeautifulSoup
from  MySQLdb.connections import Connection 
from MySQLdb.cursors import Cursor
import pandas as pd
from pandas.core.frame import DataFrame
import requests

from run import app
from config import Config


NAROU_URL = Config.NAROU_URL
NAROU_API_URL = Config.NAROU_API_URL
INTERVAL = Config.INTERVAL
SCRAPING_DETAILS_BATCH_SIZE = Config.SCRAPING_DETAILS_BATCH_SIZE


class DetailsScraper(object):
    """作品の詳細情報をスクレイピングするためのクラス
    
    Methods:
        scraping_details: 作品の詳細情報をスクレイピングするgenerator。
                                       mode='first'では最古の作品から最新の作品が対象。
                                       mode='middle'ではDB内で最新の作品から最新の作品が対象。
        __details_preprocessing: 作品の詳細情報を格納したDataFrameに対して前処理を行う。
        __get_scraping_item_number: スクレイピングの対象となる作品数を計算する。
    """

    narou_api_url = NAROU_API_URL
    interval = INTERVAL
    batch_size = SCRAPING_DETAILS_BATCH_SIZE

    @classmethod
    def get_scraped_df_iterator(cls, mode: str, conn: Connection=None, cursor: Cursor=None) -> DataFrame:
        latest_registered_datetime = cls.__get_latest_registered_datetime(mode, cursor)
        now = str(int(datetime.datetime.now().timestamp()))

        lastup = now
        allcount = cls.__get_scraping_item_number(latest_registered_datetime, now)
        all_queue_cnt = (allcount // cls.batch_size)

        for i in range(all_queue_cnt):
            payload = {
                'out': 'json', 
                'gzip': 5, 
                'opt': 'weekly',
                'lim': cls.batch_size,
                'lastup': latest_registered_datetime+"-"+str(lastup)
            }

            c = 0 # Avoid infinite loop
            while c < 5:
                try:
                    res = requests.get(cls.narou_api_url, params=payload, timeout=30).content
                    break
                except Exception as e:
                    app.logger.error(e)
                    c += 1       

            r = gzip.decompress(res).decode('utf-8')

            details_df = pd.read_json(r).drop(0)
  
            lastup = details_df.iloc[-1]["general_lastup"]
            lastup = int(datetime.datetime.strptime(lastup, "%Y-%m-%d %H:%M:%S").timestamp())

            time.sleep(cls.interval)
            yield details_df
    
    @classmethod
    def __get_latest_registered_datetime(cls, mode: str, cursor: Cursor=None) -> str:
        if mode == 'first':
            latest_registered_datetime = '1073779200'
        elif mode == 'middle' and cursor:
            cursor.execute("SELECT general_lastup FROM details ORDER BY general_lastup DESC LIMIT 1")
            sql_result = cursor.fetchone()
            latest_registered_datetime = str(sql_result[0]) if sql_result is not None else '1073779200'
        else:
            raise Exception('Argument mode should be middle or first.')

        app.logger.info(f'Latest registered datetime: {latest_registered_datetime}')
        return latest_registered_datetime

    @classmethod
    def __get_scraping_item_number(cls, latest_registered_datetime: str, now: str,) -> int:
        payload = {
            'out': 'json', 
            'gzip': 5, 
            'of': 'n',
            'lim': 1, 
            'lastup': latest_registered_datetime+"-"+now
        }
        res = requests.get(cls.narou_api_url, params=payload).content
        r =  gzip.decompress(res).decode("utf-8") 
        allcount = json.loads(r)[0]['allcount']
        return allcount

    @classmethod
    def __details_preprocessing(cls, details_df: DataFrame) -> DataFrame:
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



class TextScraper(object):
    """作品の本文をスクレイピングするためのクラス
    
    Methods:
        scraping_texts: 作品の本文をスクレイピングする。短編か連載かによって処理が異なる。
        __make_bs_obj: 引数のURLのBeautifulSoupオブジェクトを作成する。
        __get_main_text: BeautifulSoupオブジェクトを元に作品の本文をタグによって抽出する。
    """
    
    narou_url = NAROU_URL
    narou_api_url = NAROU_API_URL
    interval = INTERVAL
        
    @classmethod
    def scraping_texts(cls, ncodes: List[str]) -> Tuple[List[str], List[str]]:
        texts, processed_ncodes = [], []

        for ncode in ncodes:
            time.sleep(cls.interval)
            url = cls.narou_url + ncode + '/'
            
            c = 0 # Avoid infinite loop
            while c < 5:
                try:
                    bs_obj = cls.__make_bs_obj(url)
                    if bs_obj.findAll("dl", {"class": "novel_sublist2"}): # 連載作品の場合
                        url = cls.narou_url + ncode + '/1/'
                        bs_obj = cls.__make_bs_obj(url)
                    text = cls.__get_main_text(bs_obj)  
                    break
                except Exception as e:
                    app.logger.error(e)
                    c += 1

            processed_ncodes.append(ncode)
            texts.append(text)   
        return processed_ncodes, texts

    @classmethod
    def __make_bs_obj(cls, url: str) -> BeautifulSoup:
        html = urlopen(url)
        return BeautifulSoup(html,"html.parser")

    @classmethod
    def __get_main_text(cls, bs_obj: BeautifulSoup) -> str:
        text = ""
        text_htmls = bs_obj.findAll('div', {'id': 'novel_honbun'})[0].findAll("p")
        for text_html in text_htmls:
            text = text + text_html.get_text() + "\n"
        return text