import torch
import torch.nn as nn
from config import Config

class SentimentModel(nn.Module):
    def __init__(self, vocab_size=None):
        super().__init__()
        self.config = Config()
        
        # 詞嵌入層
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size or 10000,  # 默認詞彙表大小
            embedding_dim=self.config.EMBEDDING_DIM
        )
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=self.config.EMBEDDING_DIM,
            hidden_size=self.config.HIDDEN_DIM,
            batch_first=True
        )
        
        # 分類層
        self.fc = nn.Linear(
            self.config.HIDDEN_DIM,
            self.config.NUM_CLASSES
        )
    
    def forward(self, x):
        # x: (batch_size, sequence_length)
        x = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        _, (hidden, _) = self.lstm(x)  # hidden: (1, batch_size, hidden_dim)
        output = self.fc(hidden.squeeze(0))  # (batch_size, num_classes)
        return output
