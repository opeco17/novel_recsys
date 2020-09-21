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

    def test_scraping_and_add_good1(self):
        data = {"test": True, "mode": "first", "epoch": 1}
        response = self.__make_scraping_and_add_response(data)
        self.__test_good(response)

    def test_scraping_and_add_good2(self):
        data = {"test": True, "mode": "first", "epoch": 2}
        response = self.__make_scraping_and_add_response(data)
        self.__test_good(response)

    def test_scraping_and_add_good3(self):
        data = {"test": True, "mode": "middle", "epoch": 1}
        response = self.__make_scraping_and_add_response(data)
        self.__test_good(response)

    def test_scraping_and_add_good4(self):
        data = {"test": True, "mode": "middle", "epoch": 2}
        response = self.__make_scraping_and_add_response(data)
        self.__test_good(response)

    def test_scraping_and_add_bad1(self):
        data = {}
        response = self.__make_scraping_and_add_response(data)
        self.__test_bad(response)

    def test_scraping_and_add_bad2(self):
        data = {"test": True}
        response = self.__make_scraping_and_add_response(data)
        self.__test_bad(response)

    def test_scraping_and_add_bad3(self):
        data = {"test": True, "mode": "middle"}
        response = self.__make_scraping_and_add_response(data)
        self.__test_bad(response)

    def test_scraping_and_add_bad3(self):
        data = {"test": True, "epoch": 1}
        response = self.__make_scraping_and_add_response(data)
        self.__test_bad(response)

    def test_add_existing_data_good1(self):
        data = {"test": True, "epoch": 1}
        response = self.__make_add_existing_data_response(data)
        self.__test_good(response)

    def test_add_existing_data_good2(self):
        data = {"test": True, "epoch": 2}
        response = self.__make_add_existing_data_response(data)
        self.__test_good(response)

    def test_add_existing_data_bad1(self):
        data = {}
        response = self.__make_add_existing_data_response(data)
        self.__test_bad(response)

    def test_add_existing_data_bad2(self):
        data = {"test": True}
        response = self.__make_add_existing_data_response(data)
        self.__test_bad(response)

    def __make_scraping_and_add_response(self, data):
        response = self.client.post(
            '/scraping_and_add',
            data=json.dumps(data),
            content_type='application/json',
        )
        return response

    def __make_add_existing_data_response(self, data):
        response = self.client.post(
            '/add_existing_data',
            data=json.dumps(data),
            content_type='application/json',
        )
        return response

    def __test_good(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(json_data.get('success'))
        self.assertIsInstance(json_data.get('message'), str)


    def __test_bad(self, response):
        self.assertEqual(response.status_code, 500)
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertFalse(json_data.get('success'))
        self.assertIsInstance(json_data.get('message'), str)
        

if __name__ == '__main__':
    main()
