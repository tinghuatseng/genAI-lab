import torch
import torch.nn as nn
import torch.optim as optim
from model import SentimentModel
from data import DataProcessor
from config import Config
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self):
        self.config = Config()
        self.data_processor = DataProcessor()
        self.model = SentimentModel()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config.LEARNING_RATE
        )
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            inputs, labels = batch
            self.optimizer.zero_grad()
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self):
        # 加載數據
        train_data = self.data_processor.load_data('train')
        train_loader = self.data_processor.create_dataloader(train_data)
        
        # 訓練循環
        for epoch in range(self.config.EPOCHS):
            loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch+1}/{self.config.EPOCHS}, Loss: {loss:.4f}")
        
        # 保存模型
        torch.save(self.model.state_dict(), "sentiment_model.pth")
