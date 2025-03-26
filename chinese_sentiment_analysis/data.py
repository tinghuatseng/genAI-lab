import pandas as pd
import jieba
import re
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import torch
from config import Config

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_len=100):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 轉換為數值序列
        seq = [self.word2idx.get(word, 1) for word in text]  # 1 for UNK
        seq = seq[:self.max_len] + [0]*(self.max_len - len(seq))  # padding
        
        return torch.tensor(seq), torch.tensor(label)

class DataProcessor:
    def __init__(self):
        self.config = Config()
        self.word2idx = {}
        self.idx2word = {}
        self.stopwords = set()
        self._load_stopwords()
        
    def _load_stopwords(self):
        """加載停用詞表"""
        try:
            with open('stopwords.txt', 'r', encoding='utf-8') as f:
                self.stopwords = set([line.strip() for line in f])
        except FileNotFoundError:
            pass
            
    def _clean_text(self, text):
        """清洗文本"""
        text = re.sub(r'[^\w\s]', '', text)  # 移除標點符號
        text = re.sub(r'\s+', ' ', text)  # 移除多餘空格
        return text.strip()
        
    def build_vocab(self, texts, min_freq=5):
        """建立詞彙表"""
        word_counter = Counter()
        for text in texts:
            word_counter.update(text)
            
        # 保留頻率大於min_freq的詞
        vocab = [word for word, count in word_counter.items() 
                if count >= min_freq and word not in self.stopwords]
                
        # 添加特殊token
        vocab = ['<PAD>', '<UNK>'] + vocab
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        
    def preprocess(self, text):
        """文本預處理"""
        text = self._clean_text(text)
        words = [word for word in jieba.cut(text) if word not in self.stopwords]
        return words
        
    def load_data(self, data_type='train'):
        """加載訓練或測試數據"""
        file_path = f"{self.config.DATA_PATH}{self.config.TRAIN_FILE if data_type == 'train' else self.config.TEST_FILE}"
        df = pd.read_csv(file_path)
        
        # 預處理文本
        texts = [self.preprocess(text) for text in df['text']]
        labels = df['label'].values
        
        # 如果是訓練數據，建立詞彙表
        if data_type == 'train':
            self.build_vocab(texts)
            
        return texts, labels
    
    def create_dataloader(self, data, batch_size=None):
        """創建數據加載器"""
        texts, labels = data
        batch_size = batch_size or self.config.BATCH_SIZE
        
        dataset = TextDataset(texts, labels, self.word2idx)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
