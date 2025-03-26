import torch
from model import SentimentModel
from data import DataProcessor
from config import Config

class Predictor:
    def __init__(self, model_path="sentiment_model.pth"):
        self.config = Config()
        self.data_processor = DataProcessor()
        self.model = SentimentModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, text):
        """預測文本情感"""
        with torch.no_grad():
            # 獲取BERT輸入
            encoded_input = self.model.tokenize([text])
            
            # 前向傳播
            output = self.model(
                input_ids=encoded_input['input_ids'],
                attention_mask=encoded_input['attention_mask']
            )
            
            prediction = torch.argmax(output, dim=1).item()
            return "正面" if prediction == 1 else "負面"
