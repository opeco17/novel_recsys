import requests
from typing import List, Dict

from config import Config

class RecommendItemsGetter(object):

    @classmethod
    def get_recommend_items_by_ncode(cls, ncode: str) -> List[Dict]:
        headers = {'Content-Type': 'application/json'}
        data = {'ncode': ncode}
        response = requests.get(Config.NCODE_SEARCH_URL, headers=headers, json=data)
        if response.json().get('success'):
            recommend_items = response.json().get('recommend_items')
            for i, recommend_item in enumerate(recommend_items):
                recommend_item['url'] = f"https://ncode.syosetu.com/{recommend_item['ncode']}/"
                recommend_item['keyword'] = '#' + recommend_item['keyword'].replace(' ', ' #')
                recommend_item['rank'] = i
            return recommend_items

    
    @classmethod
    def get_recommend_items_by_text(cls, text: str) -> List[Dict]:
        headers = {'Content-Type': 'application/json'}
        data = {'text': text}
        response = requests.get(Config.TEXT_SEARCH_URL, headers=headers, json=data)
        if response.json().get('success'):
            recommend_items = response.json().get('recommend_items')
            for i, recommend_item in enumerate(recommend_items):
                recommend_item['url'] = f"https://ncode.syosetu.com/{recommend_item['ncode']}/"
                recommend_item['keyword'] = '#' + recommend_item['keyword'].replace(' ', ' #')
                recommend_item['rank'] = i
            return recommend_items