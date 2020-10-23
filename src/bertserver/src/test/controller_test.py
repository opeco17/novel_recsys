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
        self.assertIsInstance(text:=json_data.get('message'), str)

    def test_index2(self):
        response = self.client.get('/index')
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertIsInstance(text:=json_data.get('message'), str)

    def test_predict_good1(self):
        data = {'texts': 'これはテストのためのテキストです。'}
        response = self.__make_predict_response(data)
        self.__test_predict_good(response)

    def test_predict_good2(self):
        data = {'texts': ['これはテストのためのテキストです。']}
        response = self.__make_predict_response(data)
        self.__test_predict_good(response)

    def test_predict_good3(self):
        data = {'texts': ['これはテストのためのテキストです。', 'これもテストのためのテキストです。']}
        response = self.__make_predict_response(data)
        self.__test_predict_good(response)

    def test_predict_bad1(self):
        data = {'texts': 10}
        response = self.__make_predict_response(data)
        self.__test_predict_bad(response)

    def test_predict_bad2(self):
        data = {'texts': [10]}
        response = self.__make_predict_response(data)
        self.__test_predict_bad(response)
        
    def __make_predict_response(self, data):
        response = self.client.get(
            '/predict',
            data=json.dumps(data),
            content_type='application/json',
        )
        return response

    def __test_predict_good(self, response):
        self.assertEqual(response.status_code, 200)
        
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertTrue(success:=json_data.get('success'))
        self.assertIsInstance(success, bool)
        self.assertEqual(success, True)

        self.assertTrue(predictions:=json_data.get('prediction'))
        self.assertIsInstance(prediction:=predictions[0], list)
        self.assertIsInstance(prediction[0], float)
        self.assertEqual(len(prediction), app.config.get('H_DIM'))

    def __test_predict_bad(self, response):
        self.assertEqual(response.status_code, 500)
        self.assertIsInstance(json_data:=response.json, dict)
        self.assertFalse(success:=json_data.get('success'))
        

if __name__ == '__main__':
    main()