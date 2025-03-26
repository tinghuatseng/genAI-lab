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
        # 預處理文本
        processed_text = self.data_processor.preprocess(text)
        
        # 轉換為模型輸入格式
        # TODO: 實現文本到tensor的轉換
        input_tensor = torch.tensor([0])  # 佔位符
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return "正面" if prediction == 1 else "負面"
