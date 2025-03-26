import pandas as pd
from config import Config

class DataProcessor:
    def __init__(self):
        self.config = Config()
        
    def load_data(self, data_type='train'):
        """加載訓練或測試數據"""
        file_path = f"{self.config.DATA_PATH}{self.config.TRAIN_FILE if data_type == 'train' else self.config.TEST_FILE}"
        return pd.read_csv(file_path)
    
    def preprocess(self, text):
        """文本預處理"""
        # TODO: 實現中文文本預處理邏輯
        # 包括分詞、去除停用詞等
        return text
    
    def create_dataloader(self, data, batch_size=None):
        """創建數據加載器"""
        batch_size = batch_size or self.config.BATCH_SIZE
        # TODO: 實現數據加載器
        return data
