import requests

from config import Config

class RecommendItemsGetter(object):

    @classmethod
    def get_recommend_items_by_ncode(cls, ncode):
        headers = {'Content-Type': 'application/json'}
        data = {'ncode': ncode}
        response = requests.get(Config.NCODE_SEARCH_URL, headers=headers, json=data)
        if response.json().get('success'):
            recommend_items = response.json().get('recommend_items')
            for recommend_item in recommend_items:
                recommend_item['url'] = f"https://ncode.syosetu.com/{recommend_item['ncode']}/"
                recommend_item['keyword'] = '#' + recommend_item['keyword'].replace(' ', ' #')
            return recommend_items

    
    @classmethod
    def get_recommend_items_by_text(cls, text):
        headers = {'Content-Type': 'application/json'}
        data = {'text': text}
        response = requests.get(Config.TEXT_SEARCH_URL, headers=headers, json=data)
        if response.json().get('success'):
            recommend_items = response.json().get('recommend_items')
            for recommend_item in recommend_items:
                recommend_item['url'] = f"https://ncode.syosetu.com/{recommend_item['ncode']}/"
                recommend_item['keyword'] = '#' + recommend_item['keyword'].replace(' ', ' #')
            return recommend_items