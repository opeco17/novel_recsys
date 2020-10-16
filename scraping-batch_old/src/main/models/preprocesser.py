import datetime

import MeCab
from pandas.core.frame import DataFrame

from run import app


class Preprocesser(object):
    """データの前処理を行うクラス"""

    @classmethod
    def preprocess_details(cls, details_df: DataFrame) -> DataFrame:
        """DBへ追加するデータの前処理"""
        details_df = details_df.drop(['allcount', 'gensaku'], axis=1, errors='ignore')
        details_df = details_df.dropna(how='all')

        for column in details_df.columns:
            if column in ['title', 'ncode', 'userid', 'writer', 'story', 'keyword', 'text']:
                details_df[column] = details_df[column].astype(str)
            elif column in['general_firstup', 'general_lastup', 'novelupdated_at', 'updated_at']:
                details_df[column] = details_df[column].map(str).map(cls.__date_to_timestamp)
            else:
                details_df[column] = details_df[column].astype(int)

        details_df['bert_train'] =  [0 if idx % 5 == 0 else 1 for idx in details_df.index]
        details_df['ml_train'] =  [1 if idx % 5 == 0 else 0 for idx in details_df.index]
        details_df['predict_point'] = 'Nan'

        return details_df

    @classmethod
    def preprocess_ml_details(cls, details_df: DataFrame) -> DataFrame:
        """ポイント予測のために前処理を行う
        
        Variables:
            title_length: タイトルの文字数
            story_length: あらすじの文字数
            text_length: 本文の文字数
            keyword_number: キーワードの数
            noun_proportion_in_text: 本文中における文字数当たりの名詞数
        """
        mecab = MeCab.Tagger("-Ochasen")
        
        for column in ['title', 'story', 'text']:
            details_df[column + '_length'] = details_df[column].apply(lambda x: len(str(x)))
        details_df['keyword_number'] = details_df['keyword'].apply(lambda x: len(str(x).split(' ')))
        details_df['noun_proportion_in_text'] = details_df.text.apply(
                lambda x: cls.__count_noun_number(mecab, str(x)) / len(str(x))
        )
        return details_df

    @classmethod
    def __date_to_timestamp(cls, date: str) -> int:
        return int(datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S").timestamp())

    @classmethod
    def __count_noun_number(cls, mecab: MeCab.Tagger, text: str) -> int:
        count = []
        for line in mecab.parse(str(text)).splitlines():
            try:
                if "名詞" in line.split()[-1]:
                    count.append(line)
            except:
                pass
        return len(set(count))


        