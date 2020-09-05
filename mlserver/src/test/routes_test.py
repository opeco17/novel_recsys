import json
import sys
from unittest import TestCase, main
sys.path.append('../main')

import pandas as pd

from run import app


DETAILS_CSV_PATH = './test_data.csv'
USE_RECORD_NUMBER = 5


class RoutesTestCase(TestCase):

    def __init__(self, *args, **kwargs):
        super(RoutesTestCase, self).__init__(*args, **kwargs)
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.details_df = pd.read_csv(DETAILS_CSV_PATH).iloc[:USE_RECORD_NUMBER]

    def test_index1(self):
        response = self.client.get('/')
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(text:=json_data.get('message'))
        self.assertIsInstance(text, str)

    def test_index2(self):
        response = self.client.get('/index')
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(text:=json_data.get('message'))
        self.assertIsInstance(text, str)

    def test_predict_good(self):
        details_data = {
            column: list(self.details_df[column]) for column in list(self.details_df.columns)
        }
        response = self.__get_response(details_data)
        self.assertIsInstance(json_data:=response.json, dict)

        self.assertTrue(success:=json_data.get('success'))
        self.assertIsInstance(success, bool)
        self.assertEqual(success, True)

        self.assertTrue(predictions:=json_data.get('prediction'))
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(predictions[0], int)

    def test_predict_by_lack_feature(self):
        details_data = {
            column: list(self.details_df[column]) for column in list(self.details_df.columns)[:3]
        }   
        response = self.__get_response(details_data)
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertFalse(json_data.get('success'))
        self.assertTrue(message:=json_data.get('message'))
        self.assertIsInstance(message, str)

    def __get_response(self, data):
        response = self.client.post(
            '/predict',
            data=json.dumps(data),
            content_type='application/json',
        )
        return response

if __name__ == '__main__':
    main()