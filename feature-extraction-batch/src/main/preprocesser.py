import datetime

import MeCab
from MeCab import Tagger
from pandas.core.frame import DataFrame

from db_connector import DBConnector
from logger import logger


class Preprocesser(object):
    """機械学習でポイント予測を行うためのデータ前処理"""
    
    @classmethod
    def preprocess_ml_details(cls, details_df: DataFrame) -> DataFrame:
        """ポイント予測のために前処理を行う

        title_length: タイトルの文字数
        story_length: あらすじの文字数
        text_length: 本文の文字数
        keyword_number: キーワードの数
        noun_proportion_in_text: 本文中における文字数当たりの名詞数
        """
        logger.info(f"Preprocessing for point prediction start with {len(details_df)} data.")
        
        mecab = MeCab.Tagger("-Ochasen")
        
        for column in ['title', 'story', 'text']:
            details_df[column + '_length'] = details_df[column].apply(lambda x: len(str(x)))
        details_df['keyword_number'] = details_df['keyword'].apply(lambda x: len(str(x).split(' ')))
        details_df['noun_proportion_in_text'] = details_df.text.apply(
                lambda x: cls.__count_noun_number(mecab, str(x)) / len(str(x))
        )
        return details_df

    @classmethod
    def __count_noun_number(cls, mecab: Tagger, text: str) -> int:
        """本文に含まれる名詞の数をカウントする"""
        count = []
        for line in mecab.parse(str(text)).splitlines():
            try:
                if '名詞' in line.split()[-1]:
                    count.append(line)
            except Exception as e:
                extra = {'Warning': str(e)}
                logger.warning('Error occurred when count number of noun', extra=extra)
        return len(set(count))