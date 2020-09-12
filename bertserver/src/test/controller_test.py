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

    def test_predict_single_str_good(self):
        data = {'texts': 'これはテストのためのテキストです。'}
        response = self.__make_response(data)
        self.__test_predict_good(response)

    def test_predict_single_list_good(self):
        data = {'texts': ['これはテストのためのテキストです。']}
        response = self.__make_response(data)
        self.__test_predict_good(response)

    def test_predict_multiple_list_good(self):
        data = {'texts': ['これはテストのためのテキストです。', 'これもテストのためのテキストです。']}
        response = self.__make_response(data)
        self.__test_predict_good(response)

    def test_predict_bad(self):
        data = {'texts': 10}
        response = self.__make_response(data)

        self.assertIsInstance(json_data:=response.json, dict)
        self.assertFalse(success:=json_data.get('success'))

    def __test_predict_good(self, response):
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(success:=json_data.get('success'))
        self.assertIsInstance(success, bool)
        self.assertEqual(success, True)

        self.assertTrue(predictions:=json_data.get('prediction'))
        self.assertIsInstance(prediction:=predictions[0], list)
        self.assertIsInstance(prediction[0], float)
        self.assertEqual(len(prediction), app.config.get('H_DIM'))

    def __make_response(self, data):
        response = self.client.post(
            '/predict',
            data=json.dumps(data),
            content_type='application/json',
        )
        return response

if __name__ == '__main__':
    main()