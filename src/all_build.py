import argparse
import glob
import logging
import os
import subprocess
import sys


def main(args):
    """指定したディレクトリのDockerイメージをビルドする(引数が無い場合は全ディレクトリが対象)"""
    run_cmd = lambda cmd: subprocess.call(cmd, shell=True, stdout=sys.stdout, stderr=sys.stdout)
    
    dir_paths = [args.dir] if args.dir else [dockerfile_path.replace('/Dockerfile', '') for dockerfile_path in glob.glob('*/Dockerfile')]
    for dir_path in dir_paths:
        if dir_path == 'data-analysis' or (args.database == False and dir_path == 'database'):
            continue
        print(f"{dir_path} build start!")
        os.chdir(dir_path)
        run_cmd('bash build.sh')
        os.chdir('../')
        print(f"{dir_path} build finish!")
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None)
    parser.add_argument('--database', action='store_true')
    args = parser.parse_args()
    main(args)