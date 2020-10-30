import argparse

import requests

from batch_manager_connector import BatchManagerConnector
from bertserver_connector import BERTServerConnector
from config import Config
from db_connector import DBConnector
from es_connector import ElasticsearchConnector
from logger import logger
from messenger import Messenger


def execute(args: argparse.Namespace) -> None:
    test = args.test
    if test:
        logger.info(f"This is test mode.")
    try:
        conn, cursor = DBConnector.get_conn_and_cursor()
        client = ElasticsearchConnector.get_client()
        queue_data, completions = BatchManagerConnector.get_queue_data()
        ElasticsearchConnector.create_details_index_if_not_exist(client)
        details_df_iterator = DBConnector.get_details_df_iterator(conn, queue_data, completions, test)
        for i, details_df in enumerate(details_df_iterator):
            features = BERTServerConnector.extract_features(list(details_df.text))
            details_df['feature'] = features
            ElasticsearchConnector.insert_details(client, details_df)
            DBConnector.update_added_to_es(conn, cursor, details_df.ncode)
            if test and i==1:
                logger.info('Broke due to test mode.')
                break
        BatchManagerConnector.success_queue_data(queue_data)
        logger.info(f"Feature Extraction Batch of {queue_data} queue data finished!")
    
    except requests.exceptions.ConnectionError as e:
        extra = {'Error Type': type(e), 'Error': str(e)}
        logger.warning('ConnectionError occurred.', extra=extra)

    except Exception as e:
        logger.error(type(e))
        username = 'Feature Extraction Batch'
        message = 'Error occurred in feature extraction batch.'
        extra = {'Error Type': type(e), 'Error': str(e)}
        logger.error(message, extra=extra)
        Messenger.send_message(username, message)
        BatchManagerConnector.fail_queue_data(queue_data)
        
    finally:
        cursor.close()
        conn.close()
        logger.info('DB connection and cursor closed.')
        
        client.close()
        logger.info('Elasticsearch client closed.')
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    execute(args)