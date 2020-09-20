import argparse
import os
import shutil
import subprocess
import sys

def main(args):

    # Elasticsearchのデータディレクトリ処理
    if os.path.exists(es_data_old_dir:='elasticsearch/es_data_old'):
        shutil.rmtree(es_data_old_dir)
    if os.path.exists(es_data_dir:='elasticsearch/es_data'):
        os.rename(es_data_dir, es_data_old_dir)
    os.mkdir(es_data_dir)

    # Databaseのデータディレクトリ処理
    if os.path.exists(mysql_data_old_dir:='database/mysql_data_old'):
        shutil.rmtree(mysql_data_old_dir)
    if os.path.exists(mysql_data_dir:='database/mysql_data'):
        os.rename(mysql_data_dir, mysql_data_old_dir)
    os.mkdir(mysql_data_dir)

    # Dockerコマンドの実行
    run_cmd = lambda cmd: subprocess.run(cmd, shell=True, stdout=sys.stdout, stderr=sys.stdout, text=True)

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
