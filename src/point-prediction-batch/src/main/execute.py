import argparse

import requests

from batch_manager_connector import BatchManagerConnector
from db_connector import DBConnector
from config import Config
from logger import logger
from messenger import Messenger
from mlserver_connector import MLServerConnector
from preprocesser import Preprocesser


def execute(args: argparse.Namespace) -> None:
    test = args.test
    if test:
        logger.info(f"This is test mode.")
    try:
        conn, cursor = DBConnector.get_conn_and_cursor()
        queue_data, completions = BatchManagerConnector.get_queue_data()
        details_df_iterator = DBConnector.get_details_df_iterator(conn, queue_data, completions, test)
        for i, details_df in enumerate(details_df_iterator):
            details_df = Preprocesser.preprocess_ml_details(details_df)
            predicted_points = MLServerConnector.predict_point(details_df)
            DBConnector.update_predicted_point(conn, cursor, details_df.ncode, predicted_points)
            if test and i==1:
                logger.info('Broke due to test mode.')
                break
        BatchManagerConnector.success_queue_data(queue_data)
        logger.info(f"Point Prediction Batch of {queue_data} queue data finished!")
    
    except requests.exceptions.ConnectionError as e:
        extra = {'Error Type': type(e), 'Error': str(e)}
        logger.warning('ConnectionError occurred.', extra=extra)

    except Exception as e:
        logger.error(type(e))
        username = 'Point Prediction Batch'
        message = 'Error occurred in point prediction batch.'
        extra = {'Error Type': type(e), 'Error': str(e)}
        logger.error(message, extra=extra)
        Messenger.send_message(username, message)
        BatchManagerConnector.fail_queue_data(queue_data)
        
    finally:
        cursor.close()
        conn.close()
        logger.info('DB connection and cursor closed.')
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    execute(args)