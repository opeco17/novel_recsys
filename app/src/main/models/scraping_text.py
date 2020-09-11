import sys
from urllib.request import urlopen
sys.path.append('..')

from bs4 import BeautifulSoup


def scraping_text(ncode: str) -> str:
    """ncodeをクエリとして小説家になろうAPIから本文をスクレイピング"""
    base_url = 'https://ncode.syosetu.com/' + ncode
    text = None
    c = 0
    while c < 5:
        try:
            bs_obj = make_bs_obj(base_url + '/')
            if bs_obj.findAll("dl", {"class": "novel_sublist2"}): # 連載作品の場合
                bs_obj = make_bs_obj(base_url + '/1/')
            text = get_text(bs_obj)  
            break
        except Exception as e:
            print(e)
            c += 1 
    return text


def make_bs_obj(url: str) -> BeautifulSoup:
    html = urlopen(url)
    return BeautifulSoup(html, 'html.parser')


def get_text(bs_obj: BeautifulSoup) -> str:
    text = ""
    text_htmls = bs_obj.findAll('div', {'id': 'novel_honbun'})[0].findAll('p')
    for text_html in text_htmls:
        text = text + text_html.get_text() + "\n"
    return text