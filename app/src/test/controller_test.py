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
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(text:=json_data.get('text'))
        self.assertIsInstance(text, str)

    def test_index2(self):
        response = self.client.get('/index')
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(text:=json_data.get('text'))
        self.assertIsInstance(text, str)

    def test_search_by_ncode(self):
        data = {'ncode': 'n4006fe'}
        response = self.client.post(
            '/search_by_ncode', 
            data=json.dumps(data),
            content_type='application/json',
        )
        self.__test_search(response)

    def test_search_by_text(self):
        data = {'text': 'これはテストのためのテキストです。'}
        response = self.client.post(
            '/search_by_text', 
            data=json.dumps(data),
            content_type='application/json',
        )
        self.__test_search(response)

    def __test_search(self, response):
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(success:=json_data.get('success'))
        self.assertIsInstance(success, bool)
        self.assertEqual(success, True)

        self.assertTrue(recommend_items:=json_data.get('recommend_items'))
        self.assertIsInstance(recommend_items, list)

        self.assertTrue(recommend_item:=recommend_items[0])
        self.assertIsInstance(recommend_item, dict)

        self.assertTrue(ncodes:=recommend_item.get('ncode'))   
        self.assertIsInstance(ncodes[0], str)

        self.assertTrue(ncodes:=recommend_item.get('story'))   
        self.assertIsInstance(ncodes[0], str)

        self.assertTrue(ncodes:=recommend_item.get('keyword'))   
        self.assertIsInstance(ncodes[0], str)

        self.assertTrue(ncodes:=recommend_item.get('writer'))   
        self.assertIsInstance(ncodes[0], str)

if __name__ == '__main__':
    main()