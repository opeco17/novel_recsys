import os
import argparse

from scraper import Scraper


def main(args):
    if os.environ.get('HOST') == None:
        os.environ['HOST'] = args.host
    print(os.environ.get('HOST'))

    scraper = Scraper(test=args.test)
    
    mode = args.mode
    if mode == 'scraping':
        scraper.scraping_and_add()
    elif mode == 'existing':
        scraper.add_existing_data()


if __name__ == '__main__':
    """
    Args:
        host: 実行されるホストの種類(local or container)を指定。通常は指定する必要が無い。
        mode: スクレイピングの種類(scraping or existing)を指定。
              scrapingではスクレイピングを行って得られたデータを処理しDBとElasticsearchへ投入。
              existingでは既存のデータを処理しDBとElasticsearchへ投入。
        test: --testを付けるとtest=Trueとなり処理される件数が限定される。
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='local')
    parser.add_argument('--mode', type=str, default='scraping')
    parser.add_argument('--test', action='store_true')

    main(parser.parse_args())