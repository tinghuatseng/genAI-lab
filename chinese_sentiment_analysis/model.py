import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from config import Config

class SentimentModel(nn.Module):
    def __init__(self, pretrained_model='bert-base-chinese'):
        super().__init__()
        self.config = Config()
        
        # 加載預訓練的BERT模型
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, self.config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(self.config.HIDDEN_DIM, self.config.NUM_CLASSES)
        )
    
    def forward(self, input_ids, attention_mask=None):
        # 獲取BERT輸出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 使用[CLS] token的表示進行分類
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # 通過分類頭
        logits = self.classifier(pooled_output)
        return logits
    
    def tokenize(self, texts, max_length=128):
        """將文本轉換為BERT輸入格式"""
        return self.tokenizer(
            texts,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )
