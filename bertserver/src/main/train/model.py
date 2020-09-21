import math
import sys
sys.path.append('..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from transformers import AdamW, BertModel


H_DIM = Config.H_DIM
LEARNING_RATE = Config.LEARNING_RATE
NUM_CLASSES = Config.NUM_CLASSES
PRETRAINED_BERT_PATH = Config.PRETRAINED_BERT_PATH


class BERT(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.h_dim = H_DIM
        self.pretrained_bert_path = PRETRAINED_BERT_PATH
        self.bert = BertModel.from_pretrained(self.pretrained_bert_path)
        self.fc = nn.Linear(768, self.h_dim)  
    
    def forward(self, ids: torch.LongTensor, mask: torch.LongTensor) -> torch.Tensor:
        _, output = self.bert(ids, attention_mask=mask)
        output = self.fc(output)
        return output


class ArcMarginProduct(nn.Module):

    def __init__(self, s: float=30.0, m: float=0.50, device: str='cpu'):
        super(ArcMarginProduct, self).__init__()
        self.in_features = H_DIM
        self.out_features = NUM_CLASSES
        self.s = s
        self.m = m
        self.device = device

        self.weight = torch.nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=self.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine) 

        return output


class OptimizerCreater(object):

    lr = LEARNING_RATE

    @classmethod
    def create_optimizer(cls, bert: BERT, metric_fc: ArcMarginProduct) -> AdamW:
        optimizer = AdamW([
            {'params': bert.parameters(), 'lr': cls.lr},
            {'params': metric_fc.parameters(), 'lr': cls.lr},
        ])
        return optimizer