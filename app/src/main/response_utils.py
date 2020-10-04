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
        
        if not request.get_json():
            message = f"Parameter should be {self.query_is} but you throw none."
            self.__set_message_to_response_body(response_body, message)
            return response_body, status_code
        
        if not (query := request.get_json().get(self.query_is)):
            message = f"Parameter should be {self.query_is} but you throw {list(request.get_json().keys())}."
            self.__set_message_to_response_body(response_body, message)
            return response_body, status_code
        
        if not isinstance(query, str):
            message = f"Parameter {self.query_is} should be str but you throw {type(query)}."
            self.__set_message_to_response_body(response_body, message)
            return response_body, status_code
        
        self.__main_process(query, response_body)
        if not response_body['success']:
            message = "Failed similar item search."
            self.__set_message_to_response_body(response_body, message)
            return response_body, status_code
            
        status_code = 200
        message = f"search_by_{self.query_is} succeeded!"
        self.__set_message_to_response_body(response_body, message)
        return response_body, status_code
        

    def __main_process(self, query: str, response_body: Dict):
        try:
            if self.query_is == 'ncode':
                recommend_items = self.similar_item_search.similar_search_by_ncode(query)
            elif self.query_is == 'text':
                query = [query]
                recommend_items = self.similar_item_search.similar_search_by_text(query)
            
            response_body['recommend_items'] = recommend_items
            response_body['success'] = True
            
        except Exception as e:
            app.logger.error(f"Error in similar item search: {e}")
            response_body['recommend_items'] = []
            response_body['success'] = False


    def __set_message_to_response_body(self, response_body, message):
        response_body['message'] = message
        app.logger.info(message)