import json
import sys
from typing import Dict, Tuple
sys.path.append('..')

from flask import Response
from werkzeug.local import LocalProxy

from logger import logger
from models.similar_search import SimilarItemSearch


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

    def make_response_body(self, request: LocalProxy) -> Tuple[Dict, int, str]:
        """パラメータのバリデーションと類似文書検索の実行"""

        response_body = {
            'success': False,
            'recommend_items': [],
            'Content-Type': 'application/json',
        }
        status_code = 500
        
        # Validation
        if not request.get_json():
            message = f"Parameter {self.query_is} is needed but you throw none."
            return response_body, status_code, message
        
        # Parameter query_is validation
        if not (query := request.get_json().get(self.query_is)):
            message = f"Parameter {self.query_is} must be needed."
            return response_body, status_code, message
        
        if not isinstance(query, str):
            message = f"Parameter {self.query_is} should be str but you throw {type(query)}."
            return response_body, status_code, message
        
        # Parameter recommend_num validation
        if not (recommend_num := request.get_json().get('recommend_num')):
            message = f"Parameter recommend_num is needed."
            return response_body, status_code, message
        
        if not (isinstance(recommend_num, int)):
            message = f"Parameter recommend_num should be int but you throw {type(query)}."
            return response_body, status_code, message
        
        if not (recommend_num > 0):
            message = f"Parameter recommend_num should be natural number but you throw {recommend_num}."
            return response_body, status_code, message
        
        # Similar search
        self.__main_process(query, recommend_num, response_body)
        if not response_body['success']:
            message = "Failed similar item search."
            return response_body, status_code, message
            
        status_code = 200
        message = f"search_by_{self.query_is} succeeded!"
        return response_body, status_code, message
        
    def __main_process(self, query: str, recommend_num: int, response_body: Dict):
        """類似文書検索を実行"""
        try:
            if self.query_is == 'ncode':
                recommend_items = self.similar_item_search.similar_search_by_ncode(query, recommend_num)
            elif self.query_is == 'text':
                query = [query]
                recommend_items = self.similar_item_search.similar_search_by_text(query, recommend_num)
            
            response_body['recommend_items'] = recommend_items
            response_body['success'] = True
            
        except Exception as e:
            logger.error(f"Error in similar item search: {str(e)}")
            response_body['recommend_items'] = []
            response_body['success'] = False

        
def make_response(response_body: dict, status_code: int, message: str=None) -> Response:
    """レスポンスの作成"""
    if message:
        response_body['message'] = message
        logger.info(message)
    
    response = Response(
        response=json.dumps(response_body), 
        mimetype='application/json',
        status=status_code
    )
    return response