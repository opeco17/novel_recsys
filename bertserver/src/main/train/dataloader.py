import json
import os
import re
from typing import Tuple
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer

from config import Config
from train.connector import DBConnector


TRAIN_BATCH_SIZE =Config.TRAIN_BATCH_SIZE
DATASET_DIR = Config.DATASET_DIR
MAX_LENGTH = Config.MAX_LENGTH
PRETRAINED_TOKENIZER_PATH = Config.PRETRAINED_TOKENIZER_PATH


class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, df: DataFrame):
        self.max_length = MAX_LENGTH
        self.pretrained_tokenizer_path = PRETRAINED_TOKENIZER_PATH
        self.x = np.array(df['title_text'])
        self.y = np.array(df['genre_category']).astype(np.int64)
        
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(self.pretrained_tokenizer_path)
        
    def __len__(self) -> int:
        return len(self.x)
    
    def __getitem__(self, index: int) -> Tuple[torch.LongTensor, torch.LongTensor, torch.Tensor]:
        text = self.x[index]
        label = self.y[index]
        inputs = self.tokenizer.encode_plus(
            text, 
            add_special_tokens=True, 
            max_length=self.max_length,
            pad_to_max_length=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        return torch.LongTensor(ids), torch.LongTensor(mask), torch.tensor(label)


class DataLoaderCreater(object):

    batch_size = TRAIN_BATCH_SIZE
    dataset_dir = DATASET_DIR
    
    @classmethod
    def create_dataloader(cls, train: bool=True) -> DataLoader:
        df = cls.__get_df(train)
        dataset = MyDataset(df)
        dataloader = DataLoader(dataset, cls.batch_size, shuffle=True)
        return dataloader

    @classmethod
    def __get_df(cls, train: bool) -> DataFrame:
        conn, _ = DBConnector.get_conn_and_cursor()
        csv_file_name = 'train.csv' if train else 'test.csv'
        if os.path.exists(path:=os.path.join(cls.dataset_dir, csv_file_name)):
            df = pd.read_csv(path)
            latest_datetime = max(df['general_lastup'])
        else:
            df = pd.DataFrame()
            latest_datetime = 1073779200
        db_df = DBConnector.get_db_df(conn, latest_datetime, train)
        df = pd.concat([df, db_df]).reset_index(drop=True)
        conn.close()
        df = cls.__preprocess_df(df)
        return df

    @classmethod
    def __preprocess_df(cls, df: DataFrame) -> DataFrame:
        with open(os.path.join(cls.dataset_dir, 'class.json')) as f:
            genre_class_mapper = json.load(f)
        genre_class_mapper = {int(key): int(value) for key, value in genre_class_mapper.items()}
        class_list = [genre_class_mapper[genre] for genre in df.genre]
        df['genre_category'] = class_list

        title_text = []
        for i in range(len(df)):
            title_text.append(str(df.iloc[i].title) + '。' + str(df.iloc[i].text))
        df['title_text'] = title_text
        df['title_text'] = df.title_text.apply(cls.__text_preprocess)

        return df[['ncode', 'genre_category', 'title_text']]

    @classmethod
    def __text_preprocess(cls, text: str) -> str:
        text = re.sub(' ', '', text)
        text = re.sub('　', '', text)
        text = re.sub('(\n)+', '\n', text)
        text = re.sub('(\r)+', '\r', text)
        return text