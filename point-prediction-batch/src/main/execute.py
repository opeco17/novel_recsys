import argparse

from db_connector import DBConnector

from config import Config
from logger import logger
from messenger import Messenger


def execute(args: argparse.Namespace) -> None:
    test = args.test
    if test:
        logger.info(f"This is test mode.")
        
    

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    execute(args)