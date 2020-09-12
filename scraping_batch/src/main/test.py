from models.scraper import Scraper
from run import logger

logger.info('test')

# scraper = Scraper(mode='first', test=True)
# scraper.scraping_and_add()

scraper = Scraper(mode='middle', test=True)
scraper.add_existing_data()