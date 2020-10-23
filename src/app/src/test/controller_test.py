import json
import sys
from unittest import TestCase, main
sys.path.append('../main')

from run import app


class ControllerTestCase(TestCase):

    def __init__(self, *args, **kwargs):
        super(ControllerTestCase, self).__init__(*args, **kwargs)
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index1(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertIsInstance(json_data.get('message'), str)

    def test_index2(self):
        response = self.client.get('/index')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertIsInstance(json_data.get('message'), str)

    def test_search_by_ncode_good1(self):
        data = {'ncode': 'n4006fe'}
        response = self.__make_test_search_by_ncode_response(data)
        self.__test_search_good(response)

    def test_search_by_ncode_good2(self):
        data = {'ncode': 'N4006FE'}
        response = self.__make_test_search_by_ncode_response(data)
        self.__test_search_good(response)

    def test_search_by_ncode_bad1(self):
        data = {}
        response = self.__make_test_search_by_ncode_response(data)
        self.__test_search_bad(response)

    def test_search_by_ncode_bad2(self):
        data = {'ncode': 1}
        response = self.__make_test_search_by_ncode_response(data)
        self.__test_search_bad(response)

    def test_search_by_text_good(self):
        data = {'text': 'これはテストのためのテキストです。'}
        response = self.__make_test_search_by_text_response(data)
        self.__test_search_good(response)

    def test_search_by_text_bad1(self):
        data = {}
        response = self.__make_test_search_by_text_response(data)
        self.__test_search_bad(response)

    def test_search_by_text_bad2(self):
        data = {'text': 1}
        response = self.__make_test_search_by_text_response(data)
        self.__test_search_bad(response)

    def __test_search_good(self, response):
        self.assertEqual(response.status_code, 200)
        
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(json_data.get('success'))

        self.assertIsInstance(recommend_items:=json_data.get('recommend_items'), list)
        self.assertIsInstance(recommend_item:=recommend_items[0], dict)

        self.assertIsInstance(recommend_item.get('ncode'), str)    
        self.assertIsInstance(recommend_item.get('story'), str)
        self.assertIsInstance(recommend_item.get('keyword'), str)
        self.assertIsInstance(recommend_item.get('writer'), str) 
        self.assertIsInstance(recommend_item.get('genre'), int)
        self.assertIsInstance(recommend_item.get('biggenre'), int)

    def __test_search_bad(self, response):
        self.assertEqual(response.status_code, 500)
        
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertFalse(json_data.get('success'))

        self.assertIsInstance(json_data.get('message'), str)

    def __make_test_search_by_ncode_response(self, data):
        response = self.client.get(
            '/search_by_ncode', 
            data=json.dumps(data),
            content_type='application/json',
        )
        return response

    def __make_test_search_by_text_response(self, data):
        response = self.client.get(
            '/search_by_text', 
            data=json.dumps(data),
            content_type='application/json',
        )
        return response

if __name__ == '__main__':
    main()