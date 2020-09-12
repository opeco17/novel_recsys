import sys
from typing import List
sys.path.append('..')

import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertModel

from config import Config
from run import app

H_DIM = Config.H_DIM
MAX_LENGTH = Config.MAX_LENGTH
PARAMETER_PATH = Config.PARAMETER_PATH
PRETRAINED_BERT_PATH = Config.PRETRAINED_BERT_PATH
PRETRAINED_TOKENIZER_PATH = Config.PRETRAINED_TOKENIZER_PATH


class BERT(nn.Module):
    
    def __init__(self):
        """学習済みBERTモデルのアーキテクチャと特徴量抽出のための全結合層を定義"""
        super().__init__()
        self.bert = BertModel.from_pretrained(PRETRAINED_BERT_PATH)
        self.fc = nn.Linear(768, H_DIM)
    
    def forward(self, ids: torch.LongTensor, mask: torch.LongTensor) -> torch.FloatTensor:
        """ネットワークの処理フローを定義

        Args:
            ids: 文書のidベクトル (Tokenizerに文書を入力することで取得)
            mask: 文書のmaskベクトル (Tokenizerに文書を入力することで取得)
        Returns:
            output: 文書の特徴量ベクトル
        """
        _, output = self.bert(ids, attention_mask=mask)
        output = self.fc(output)
        return output


class FeatureExtractor(object):

    def __init__(self):
        """特徴量抽出に関係するフィールドの定義

        Args:
            h_dim: 特徴量の次元数
            max_length: BERTで処理する文書の最大の長さ
            parameter_path: ArcFaceで学習を行った学習済みモデルのパラメータのパス
            pretrained_bert_path: 東北大学の学習済みBERTモデルのパラメータのパス
            pretrained_tokenizer_path: 東北大学によって作成されたBERTを使用するためのTokenizerのパス
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_length = MAX_LENGTH
        self.bert = BERT().to(self.device)
        self.bert.load_state_dict(torch.load(PARAMETER_PATH, map_location=self.device))
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(PRETRAINED_TOKENIZER_PATH)
        app.logger.info('FeatureExtractor constructed!')
        
    
    def extract(self, texts: List[str]) -> List[float]:
        """文書から特徴量の抽出を行うメソッド

        入力文書をTokenizerで変換してidsベクトルとmaskベクトルを得る。
        2つのベクトルをBERTで処理し特徴量抽出を行う。
        """
        ids, masks = [], []
        for text in texts:
            input = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                pad_to_max_length=True,
                truncation=True
            )
            ids.append(input['input_ids'])
            masks.append(input['attention_mask'])
        
        with torch.no_grad():
            ids_tensor = torch.LongTensor(ids).to(self.device)
            masks_tensor = torch.LongTensor(masks).to(self.device)
            outputs = self.bert(ids_tensor, masks_tensor)
            outputs = outputs.tolist()
        
        return outputs