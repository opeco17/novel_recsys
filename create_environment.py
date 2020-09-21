# coding: utf-8

import argparse
import os
import shutil
import subprocess
import sys

def main(args):

    # Elasticsearchのデータディレクトリ処理
    es_data_old_dir = 'elasticsearch/es_data_old'
    es_data_dir = 'elasticsearch/es_data'
    if os.path.exists(es_data_old_dir):
        shutil.rmtree(es_data_old_dir)
    if os.path.exists(es_data_dir):
        os.rename(es_data_dir, es_data_old_dir)
    os.makedirs(es_data_dir)

    # Databaseのデータディレクトリ処理
    mysql_data_old_dir = 'database/mysql_data_old'
    mysql_data_dir = 'database/mysql_data'
    if os.path.exists(mysql_data_old_dir):
        shutil.rmtree(mysql_data_old_dir)
    if os.path.exists(mysql_data_dir):
        os.rename(mysql_data_dir, mysql_data_old_dir)
    os.makedirs(mysql_data_dir)

    # Dockerコマンドの実行
    run_cmd = lambda cmd: subprocess.call(cmd, shell=True, stdout=sys.stdout, stderr=sys.stdout)

    run_cmd('docker network create narou_network')

    print(args.build)
    if args.build:
        print('Docker build start.')
        run_cmd("docker-compose up -d --build")
    else:
        print('Docker build not start.')
        run_cmd('docker-compose up -d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Dockerのbuildを行うか指定')
    args = parser.parse_args()
    main(args)
