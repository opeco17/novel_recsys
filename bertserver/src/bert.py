import torch
import torch.nn as nn
from transformers import BertJapaneseTokenizer, BertModel



class BERT(nn.Module):
    
    def __init__(self, pretrained_bert_path, h_dim):
        '''
        Args:
            str pretrained_bert_path: path of bert model pretrained by Tohoku univ.
            int h_dim: number of nodes in last hidden layer
        '''
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_bert_path)
        self.fc = nn.Linear(768, h_dim)
        
    
    def forward(self, ids, mask):
        '''
        Args:
            torch.TensorLong ids: id vectors of text
            torch.TensorLong mask: mask vectors of text
        Returns:
            torch.Tensor output: feature vectors of text
        '''
        _, output = self.bert(ids, attention_mask=mask)
        output = self.fc(output)
        return output



class FeatureExtractor(object):

    def __init__(self, h_dim, max_length, parameter_path, pretrained_bert_path, pretrained_tokenizer_path):
        '''
        Args:
            int h_dim : number of nodes in last hidden layer
            str max_length: length of text used for prediction and training
            str parameter_path: model parameter path tranined by narou texts
            str pretrained_bert_path: path of bert model pretrained by Tohoku univ.
            str pretrained_tokenizer_path: path of bert tokenizer pretrained by Tohoku univ.
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_length = max_length
        self.bert = BERT(pretrained_bert_path, h_dim).to(self.device)
        self.bert.load_state_dict(torch.load(parameter_path, map_location=self.device))

        self.tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_tokenizer_path)

    
    def extract(self, texts):
        '''
        Args:
            list<str> texts: texts list of narou novel
        Returns:
            list<float> outputs: feature vectors list
        '''
        if type(texts) != list:
            raise Exception('extract method takes only list of text.')
        
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



def load_model(h_dim, max_length, parameter_path, pretrained_bert_path, pretrained_tokenizer_path):
    model = FeatureExtractor(h_dim, max_length, parameter_path, pretrained_bert_path, pretrained_tokenizer_path)
    return model