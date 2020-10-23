import time
from typing import List
from urllib.request import urlopen

from bs4 import BeautifulSoup

from config import Config
from logger import logger


NAROU_URL = Config.NAROU_URL
SCRAPING_INTERVAL = Config.SCRAPING_INTERVAL


class TextScraper(object):
    """作品の本文をスクレイピングする"""
    
    @classmethod
    def scrape_texts(cls, ncodes: List[str]):
        """受け取ったncodeに対応する本文をスクレイピングして提供"""
        texts = []
        
        for ncode in ncodes:
            time.sleep(SCRAPING_INTERVAL)
            url = NAROU_URL + ncode + '/'
           
            c = 0
            while True:
                try:
                    bs_obj = cls.__make_bs_obj(url)
                    # 連載作品の場合には処理を挟む
                    if bs_obj.findAll("dl", {"class": "novel_sublist2"}):
                        url = NAROU_URL + ncode + '/1/'
                        bs_obj = cls.__make_bs_obj(url)
                    # 本文を取得
                    text = cls.__scrape_main_text(bs_obj) 
                    break
                except Exception as e:
                    extra = {'Class': 'TextScraper', 'Method': 'scrape_texts', 'Error': str(e)}
                    logger.error('Unable to scrape main text.', extra=extra)
                    c += 1
                    if c == 5:
                        raise
            
            texts.append(text)
        return texts
            
    @classmethod
    def __make_bs_obj(cls, url: str) -> BeautifulSoup:
        """本文のスクレイピングのためにBeatifulSoupオブジェクトを作成"""
        html = urlopen(url)
        return BeautifulSoup(html, 'html.parser')
    
    @classmethod
    def __scrape_main_text(cls, bs_obj: BeautifulSoup) -> str:
        """本文をスクレイピングする"""
        text = ''
        text_htmls = bs_obj.findAll('div', {'id': 'novel_honbun'})[0].findAll('p')
        for text_html in text_htmls:
            text = text + text_html.get_text() + '\n'
        return text