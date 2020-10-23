import argparse

from db_connector import DBConnector
from details_scraper import DetailsScraper
from preprocesser import Preprocesser
from text_scraper import TextScraper

from config import Config
from logger import logger
from messenger import Messenger


def execute(args: argparse.Namespace) -> None:
    test = args.test
    if test:
        logger.info(f"This is test mode.")
        
    try:
        conn, cursor = DBConnector.get_conn_and_cursor()
        details_df_iterator = DetailsScraper.get_details_df_iterator(test, cursor)
        for i, sub_details_df in enumerate(details_df_iterator):
            texts = TextScraper.scrape_texts(sub_details_df.ncode)
            texts = [text[:1024] for text in texts]
            sub_details_df['text'] = texts
            sub_details_df = Preprocesser.preprocess_details(sub_details_df)
            DBConnector.insert_details(conn, cursor, sub_details_df)
        
        username = 'Scraping batch'
        message = 'Scraping batch completed!'
        logger.info(message)
        Messenger.send_message(username, message)
        
    except Exception as e:
        username = 'Scraping batch'
        message = 'Error occurred in scraping batch.'
        extra = {'Error': str(e)}
        logger.error(message, extra=extra)
        Messenger.send_message(username, message)
        
    finally:
        cursor.close()
        conn.close()
        logger.info('DB connection and cursor closed.')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    execute(args)