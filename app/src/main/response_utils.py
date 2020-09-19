import json
import sys
from typing import Dict, Tuple
sys.path.append('..')

from models.similar_search import SimilarItemSearch
from werkzeug.local import LocalProxy

from run import app


class ResponseMakerForNcodeAndText(object):
    """Responseを生成するクラス

    類似文書検索のクエリがncodeまたはtextの場合にエラーハンドリングを行いresponseを生成する。

    Fields:
        query_is: クエリがncodeの場合は'ncode'、textの場合は'text'を指定。
        similar_search: クエリを元に類似文書検索を行いレコメンドリストを返す。
    """

    def __init__(self, query_is: str):
        self.query_is = query_is
        self.similar_item_search = SimilarItemSearch()

    def make_response_body(self, request: LocalProxy) -> Tuple[Dict, int]:

        response_body = {
            'success': False,
            'recommend_items': [],
            'Content-Type': 'application/json',
        }
        status_code = 500

        if request.get_json():
            if query := request.get_json().get(self.query_is):
                if isinstance(query, str):
                    self.__main_process(query, response_body)
                    status_code = 200
                else:
                    response_body['message'] = f'Parameter {self.query_is} should be str but you throw {type(query)}.'
            else:
                keys = list(request.get_json().keys())
                response_body['message'] = f'Parameter should be {self.query_is} but you throw {keys}.'
        else:
            response_body['message'] = f'Parameter should be {self.query_is} but you throw none.'

        return response_body, status_code

    def __main_process(self, query: str, response_body: Dict):
        if self.query_is == 'ncode':
            recommend_items = self.similar_item_search.similar_search_by_ncode(query)
        elif self.query_is == 'text':
            query = [query]
            recommend_items = self.similar_item_search.similar_search_by_text(query)
        
        response_body['recommend_items'] = recommend_items
        response_body['success'] = True
