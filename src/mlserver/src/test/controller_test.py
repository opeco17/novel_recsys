import json
import sys
from unittest import TestCase, main
sys.path.append('../main')

import pandas as pd

from run import app

DETAILS_CSV_PATH = './test_data.csv'
USE_RECORD_NUMBER = 5


class ControllerTestCase(TestCase):

    def __init__(self, *args, **kwargs):
        super(ControllerTestCase, self).__init__(*args, **kwargs)
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.details_df = pd.read_csv(DETAILS_CSV_PATH).iloc[:USE_RECORD_NUMBER]

    def test_index1(self):
        response = self.client.get('/')
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertIsInstance(json_data.get('message'), str)

    def test_index2(self):
        response = self.client.get('/index')
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertIsInstance(json_data.get('message'), str)

    def test_predict_good(self):
        details_data = {
            column: list(self.details_df[column]) for column in list(self.details_df.columns)
        }
        response = self.__make_predict_response(details_data)
        self.assertEqual(response.status_code, 200)

        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(success:=json_data.get('success'))
        self.assertTrue(success)

        self.assertTrue(predictions:=json_data.get('prediction'))
        self.assertIsInstance(predictions, list)
        self.assertIsInstance(predictions[0], int)

    def test_predict_bad(self):
        details_data = {
            column: list(self.details_df[column])[:USE_RECORD_NUMBER] for column in list(self.details_df.columns)[:3]
        }   
        response = self.__make_predict_response(details_data)
        self.assertEqual(response.status_code, 500)

        self.assertIsInstance(json_data:=response.json, dict)
        self.assertFalse(json_data.get('success'))
        self.assertIsInstance(json_data.get('message'), str)

    def __make_predict_response(self, data):
        response = self.client.get(
            '/predict',
            data=json.dumps(data),
            content_type='application/json',
        )
        return response


if __name__ == '__main__':
    main()