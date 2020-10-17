import datetime
import gzip
import json
import random
import time
from urllib3.util import Retry

from bs4 import BeautifulSoup
from MySQLdb.cursors import Cursor
import pandas as pd
from pandas.core.frame import DataFrame
import requests
from requests.adapters import HTTPAdapter
from requests.sessions import Session

from config import Config
from logger import logger
from text_scraper import TextScraper


SCRAPING_INTERVAL = Config.SCRAPING_INTERVAL
SCRAPING_DETAILS_BATCH_SIZE = Config.SCRAPING_DETAILS_BATCH_SIZE
NAROU_API_URL = Config.NAROU_API_URL


class DetailsScraper(object):
    """作品の情報をスクレイピングする"""

    @classmethod
    def get_details_df_iterator(cls, test: bool, cursor: Cursor) -> DataFrame:
        """作品情報のミニバッチのイテレータを提供"""
        latest_registered_datetime = cls.__get_latest_registered_datetime(test, cursor)
        now = str(int(datetime.datetime.now().timestamp()))
        lastup = now
        all_count = cls.__get_scraping_item_number(latest_registered_datetime, now)
        if test:
            all_queue_count = Config.ITERATION_OF_TEST
        else:
            all_queue_count = all_count // SCRAPING_DETAILS_BATCH_SIZE
        
        for i in range(all_queue_count):
            logger.info(f"Scraping status: {i+1} / {all_queue_count}")
            sub_details_df = cls.__scrape_sub_details_df(latest_registered_datetime, lastup)
            lastup = sub_details_df.iloc[-1].get('general_lastup')
            lastup = int(datetime.datetime.strptime(lastup, "%Y-%m-%d %H:%M:%S").timestamp())
            time.sleep(SCRAPING_INTERVAL)
            yield sub_details_df
    
    @classmethod
    def __get_latest_registered_datetime(cls, test: bool, cursor: Cursor) -> str:
        """DBに登録されている最新のUNIX時刻を取得する (テストの際はランダムに取得)"""
        if test:
            # 2004年1月11日から2011年1月11日の間からランダムに取得
            latest_registered_datetime = str(random.randint(1073779200, 1168484400))
        else:
            # DBに登録されている最新の時刻を取得
            try:
                cursor.execute("SELECT general_lastup FROM details ORDER BY general_lastup DESC LIMIT 1")
                result = cursor.fetchone()
                latest_registered_datetime = str(result[0]) if result is not None else '1073779200'
            except Exception as e:
                extra = {'Class': 'DetailsScraper', 'Method': '__get_latest_registered_datetime', 'Error': e}
                logger.error('Exception occured.', extra=extra)
                raise
        
        logger.info(f"Latest registered datetime: {latest_registered_datetime}")
        return latest_registered_datetime
        
    @classmethod
    def __get_scraping_item_number(cls, latest_registered_datetime: str, now: str,) -> int:
        """スクレイピングする作品の総数を取得する"""
        payload = {'out': 'json', 'gzip': 5, 'of': 'n', 'lim': 1, 'lastup': latest_registered_datetime+"-"+now}
        session = cls.__get_retry_session()
        try:
            response = session.get(url=NAROU_API_URL, params=payload, timeout=15)
        except requests.exceptions.ConnectTimeout:
            extra = {'Class': 'DetailsScraper', 'Method': '__get_scraping_item_number'}
            logger.error('Unable to get scraping_item_number')
        except Exception as e:
            extra = {'Class': 'DetailsScraper', 'Method': '__get_scraping_item_number', 'Error': e}
            logger.error('Unable to get scraping_item_number')
            raise
            
        all_count = json.loads(gzip.decompress(response.content).decode('utf-8') )[0]['allcount']
        return all_count
    
    @classmethod
    def __scrape_sub_details_df(cls, latest_registered_datetime: str, lastup: str) -> DataFrame:
        """作品の詳細情報のミニバッチをスクレイピングする"""
        payload = {
            'out': 'json', 'gzip': 5, 'opt': 'weekly','lim': SCRAPING_DETAILS_BATCH_SIZE, \
            'lastup': f"{latest_registered_datetime}-{lastup}"
        }
        session = cls.__get_retry_session()
        try:
            response = session.get(url=NAROU_API_URL, params=payload, timeout=15)
        except requests.exceptions.ConnectTimeout:
            extra = {'Class': 'DetailsScraper', 'Method': '__scrape_sub_details'}
            logger.error('Unable to get details.')
        except Exception as e:
            extra = {'Class': 'DetailsScraper', 'Method': '__scrape_sub_details', 'Error': e}
            logger.error('Unable to get details.')
            raise
            
        sub_details_df = pd.read_json(gzip.decompress(response.content).decode('utf-8')).drop(0)
        return sub_details_df
    
    @classmethod
    def __get_retry_session(cls) -> Session:
        """リクエストが失敗した時にリトライを行うためのセッションを提供"""
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session