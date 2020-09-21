import json
import sys
from flask import Flask
from flask_testing import TestCase
from unittest import main
sys.path.append('../main')

from run import app


class ControllerTestCase(TestCase):

    def create_app(self):
        app.config['TESTING'] = True
        return app

    def test_index1(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assert_template_used('index.html')

    def test_index2(self):
        response = self.client.get('/index')
        self.assertEqual(response.status_code, 200)
        self.assert_template_used('index.html')

    def test_search_by_text_get(self):
        response = self.client.get('/search_by_text')
        self.assertEqual(response.status_code, 200)
        self.assert_template_used('search_by_text.html')

    def test_search_by_text_post_good(self):
        response = self.client.post('/search_by_text', data={'text': 'これはテストのためのテキストです。'})
        self.assertEqual(response.status_code, 200)
        self.assert_template_used('search_by_text.html')

    def test_search_by_url_post_good(self):
        response = self.client.post('/search_by_text', data={'url': 'https://ncode.syosetu.com/N6755gk/'})
        self.assertEqual(response.status_code, 200)
        self.assert_template_used('search_by_text.html')


if __name__ == '__main__':
    main()
