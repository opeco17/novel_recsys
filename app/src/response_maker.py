from similar_search import SimilarTextSearch

class ResponseMakerForNcodeAndText(object):
    """Responseを生成するクラス

    類似文書検索のクエリがncodeまたはtextの場合にエラーハンドリングを行いresponseを生成する。

    Fields:
        query_is: クエリがncodeの場合は'ncode'、textの場合は'text'を指定。
        similar_search: クエリを元に類似文書検索を行いレコメンドリストを返す。
    """

    def __init__(self, query_is):
        self.query_is = query_is
        self.similar_text_search = SimilarTextSearch()

    def make_response(self, request):

        response = {
            'success': False,
            'recommend_ncodes': [],
            'Content-Type': 'application/json',
        }

        if request.method == 'POST':
            if request.get_json():
                if query := request.get_json().get(self.query_is):
                    if (query_type := type(query)) is str:
                        self._main_process(query, response)
                    else:
                        response['message'] = f'Parameter {self.query_is} should be str but you throw {query_type}.'
                else:
                    keys = list(request.get_json().keys())
                    response['message'] = f'Parameter should be {self.query_is} but you throw {keys}.'
            else:
                response['message'] = f'Parameter should be {self.query_is} but you throw none.'

        return response

    def _main_process(self, query, response):
        if self.query_is == 'ncode':
            recommend_ncodes = self.similar_text_search.similar_search_by_ncode(query)
        elif self.query_is == 'text':
            query = [query]
            recommend_ncodes = self.similar_text_search.similar_search_by_text(query)
        
        response['recommend_ncodes'] = recommend_ncodes
        response['success'] = True
