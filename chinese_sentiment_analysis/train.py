import torch
import torch.nn as nn
import torch.optim as optim
from model import SentimentModel
from data import DataProcessor
from config import Config
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer:
    def __init__(self):
        self.config = Config()
        self.data_processor = DataProcessor()
        self.model = SentimentModel()
        self.criterion = nn.CrossEntropyLoss()
        
        # 使用AdamW優化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            eps=1e-8
        )
        
        # 學習率調度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.config.EPOCHS * 1000  # 假設每epoch有1000步
        )
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            inputs, labels = batch
            self.optimizer.zero_grad()
            
            # 獲取BERT輸入
            encoded_inputs = self.model.tokenize(inputs)
            
            # 前向傳播
            outputs = self.model(
                input_ids=encoded_inputs['input_ids'],
                attention_mask=encoded_inputs['attention_mask']
            )
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self):
        # 加載數據
        train_data, train_labels = self.data_processor.load_data('train')
        val_data, val_labels = self.data_processor.load_data('test')
        
        train_loader = self.data_processor.create_dataloader((train_data, train_labels))
        val_loader = self.data_processor.create_dataloader((val_data, val_labels))
        
        best_val_loss = float('inf')
        patience = 3
        no_improve = 0
        
        for epoch in range(self.config.EPOCHS):
            # 訓練階段
            train_loss = self.train_epoch(train_loader)
            
            # 驗證階段
            val_loss = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # 早停機制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 加載最佳模型
        self.model.load_state_dict(torch.load("best_model.pth"))
        torch.save(self.model.state_dict(), "sentiment_model.pth")
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                encoded_inputs = self.model.tokenize(inputs)
                
                outputs = self.model(
                    input_ids=encoded_inputs['input_ids'],
                    attention_mask=encoded_inputs['attention_mask']
                )
                
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
