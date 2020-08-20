import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertModel



H_DIM = 64
MAX_LENGTH = 512
PRETRAINED_BERT_PATH = 'parameters/pretrained_bert'
PRETRAINED_TOKENIZER_PATH = 'parameters/pretrained_tokenizer'
PARAMETER_PATH = 'parameters/bert40000.pth'



class BERT(nn.Module):
    
    def __init__(self, pretrained, h_dim):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained)
        self.fc = nn.Linear(768, h_dim)
        
    
    def forward(self, ids, mask):
        _, output = self.bert(ids, attention_mask=mask)
        output = self.fc(output)
        return output



class FeatureExtractor(object):

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.bert = BERT(PRETRAINED_BERT_PATH, H_DIM).to(self.device)
        self.bert.load_state_dict(torch.load(PARAMETER_PATH, map_location=self.device))

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(PRETRAINED_TOKENIZER_PATH)

    
    def extract(self, texts):
        if type(texts) != list:
            raise Exception('extract method takes only list of text.')
        
        ids, masks = [], []
        for text in texts:
            input = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
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



def load_model():
    model = FeatureExtractor()
    return model