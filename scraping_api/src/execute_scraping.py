import argparse

from scraper import Scraper


def main(args):
    """
    実行時に--testを付けるとtest=Trueとなり、付けない場合はFalseとなる。
    """
    mode = args.mode
    scraper = Scraper(test=args.test)
    if mode == 'scraping':
        scraper.scraping_and_add()
    elif mode == 'existing':
        scraper.add_existing_data()
    else:
        raise Exception("Argument mode should be 'scraping' or 'existing'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='scraping')
    parser.add_argument('--test', action='store_true')

    main(parser.parse_args())