import json
import time
import datetime
import gzip
from urllib.request import urlopen

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from utils import *


TEST = True
MINIBATCH_SIZE = 10
NAROU_API_URL = 'https://api.syosetu.com/novelapi/api/'
ELASTICSEARCH_HOST_NAME = 'elasticsearch'
FEATURE_EXTRACTION_URL = 'http://bertserver:3032/predict'
POINT_PREDICTION_URL = 'http://mlserver:3033/predict'
DB_HOST_NAME = 'db'
# ELASTICSEARCH_HOST_NAME = 'localhost:9200'
# FEATURE_EXTRACTION_URL = 'http://localhost:3032/predict'
# POINT_PREDICTION_URL = 'http://localhost:3033/predict'
# DB_HOST_NAME = '0.0.0.0'


# Scraping detail of items

def _details_preprocessing(df):
    df = df.drop(['allcount', 'gensaku'], axis=1, errors='ignore')
    df = df.dropna(how='all')
    
    date_to_timestamp = lambda date: int(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp())

    for column in df.columns:
        if column in ['title', 'ncode', 'userid', 'writer', 'story', 'keyword']:
            df[column] = df[column].astype(str)
        elif column in['general_firstup', 'general_lastup', 'novelupdated_at', 'updated_at']:
            df[column] = df[column].map(str).map(date_to_timestamp)
        else:
            df[column] = df[column].astype(int)
            
    df['predict_point'] = 'Nan'
    df['text'] = 'Nan'
    
    return df


def scraping_details(conn, cursor, narou_api_url, mode='middle', test=True):
    if mode not in ['middle', 'first']:
        raise Exception('Argument mode should be middle or first.')
    
    if mode == 'middle':
        cursor.execute('SELECT general_lastup FROM details ORDER BY general_lastup DESC LIMIT 1')
        sql_result = cursor.fetchone()
        register_latest = str(sql_result[0]) if sql_result is not None else "1073779200"
    elif mode == 'first':
        register_latest = "1073779200"
    
    now = str(int(datetime.datetime.now().timestamp()))

    payload = {'out': 'json', 'gzip': 5, 'of': 'n', 'lim': 1, 'lastup': register_latest+"-"+now}
    res = requests.get(narou_api_url, params=payload).content
    r =  gzip.decompress(res).decode("utf-8") 
    allcount = json.loads(r)[0]["allcount"]
    
    interval = 1
    details_df = pd.DataFrame()

    lastup = now
    all_queue_cnt = (allcount // 500)

    for i in range(all_queue_cnt):
        payload = {'out': 'json', 'gzip': 5,'opt': 'weekly', 'lim':500, 'lastup': register_latest+"-"+str(lastup)}
        
        c = 0 # Avoid infinite loop
        while c < 10:
            try:
                res = requests.get(narou_api_url, params=payload, timeout=30).content
                break
            except:
                print('Connection Error')
                c += 1       

        r = gzip.decompress(res).decode('utf-8')

        df_temp = pd.read_json(r)
        df_temp = df_temp.drop(0)

        last_general_lastup = df_temp.iloc[-1]["general_lastup"]
        lastup = datetime.datetime.strptime(last_general_lastup, "%Y-%m-%d %H:%M:%S").timestamp()
        lastup = int(lastup)

        df_temp = _details_preprocessing(df_temp)
        details_df = pd.concat([details_df, df_temp], axis=0)

        time.sleep(interval)
        
        if test is True and i==1:
            break
        
    return details_df



# Scraping text of items

def _make_bs_obj(url):
    html = urlopen(url)
    return BeautifulSoup(html,"html.parser")



def _get_main_text(bs_obj):
    text = ""
    text_htmls = bs_obj.findAll("div",{"id":"novel_honbun"})[0].findAll("p")

    for text_html in text_htmls:
        text = text + text_html.get_text() + "\n\n"

    return text



def scraping_texts(ncodes, test=True):
    texts = []
    processed_ncodes = []
    interval = 0.1
    cnt = 0

    for ncode in ncodes:
        print(cnt) if cnt % 100 == 0 else None

        time.sleep(interval)
        url = 'https://ncode.syosetu.com/' + ncode + '/'
        c = 0 # Avoid infinite loop
        while c < 10:
            try:
                bs_obj = _make_bs_obj(url)
                break
            except:
                print('Connection Error')
                c += 1
                
        url_list = ["https://ncode.syosetu.com" + a_bs_obj.find("a").attrs["href"] for a_bs_obj in bs_obj.findAll("dl", {"class": "novel_sublist2"})]
        
        if len(url_list) == 0:
            text = _get_main_text(bs_obj)
        else:
            time.sleep(interval)
            bs_obj = _make_bs_obj(url_list[0])
            text = _get_main_text(bs_obj)

        texts.append(text)
        processed_ncodes.append(ncode)
        cnt += 1
        
        if test == True and cnt == 10:
            break
    
    return processed_ncodes, texts



def register_scraped_data():
    conn, cursor = get_connector_and_cursor(DB_HOST_NAME)
    # Scraping details and texts
    detail_df = scraping_details(conn, cursor, NAROU_API_URL, mode='first', test=TEST)
    ncodes, texts = scraping_texts(detail_df.ncode, TEST)
    for ncode, text in zip(ncodes, texts):
        detail_df.loc[detail_df['ncode'] == ncode, 'text'] = text
    predicted_point = point_prediction(POINT_PREDICTION_URL, detail_df)
    detail_df['predict_point'] = predicted_point
    
    # Insert scraped data to database
    cursor.execute('SHOW columns FROM details')
    columns_of_details = [column[0] for column in cursor.fetchall()]
    details_data_tmp = detail_df[columns_of_details]
    details_data = [tuple(details_data_tmp.iloc[i]) for i in range(len(details_data_tmp))]
    cursor.executemany("INSERT INTO details VALUES ({})".format(("%s, "*len(columns_of_details))[:-2]), details_data)
    
    # Insert scraped data to elasticsearch
    target_detail_df = detail_df[(detail_df['predict_point'] == 1) & (detail_df['global_point'] == 0)]
    if len(target_detail_df) != 0:      
        ncodes = list(target_detail_df.ncode)
        texts = list(target_detail_df.text)

        for i in range(len(ncodes) // MINIBATCH_SIZE + 1):
            register_features_to_elasticsearch(ELASTICSEARCH_HOST_NAME, FEATURE_EXTRACTION_URL, ncodes[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE], texts[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE])
            if TEST:
                break
    
    print('{} data is inserted to Elasticsearch.'.format(len(target_detail_df)))
    conn.commit()
    conn.close()



if __name__ == '__main__':
    register_scraped_data()