import json
import sys
from unittest import TestCase, main
sys.path.append('../main')

from models.scraper import Scraper


class ScraperTestCase(TestCase):

    def __init__(self, *args, **kwargs):
        super(ScraperTestCase, self).__init__(*args, **kwargs)
        self.scraper = Scraper(test=True)

    def scraping_and_add_test(self):
         self.scraper.scraping_and_add()

    def add_existing_data_test(self):
        self.scraper.add_existing_data()


if __name__ == '__main__':
    main()
