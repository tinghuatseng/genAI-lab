# 配置文件
class Config:
    # 數據路徑配置
    DATA_PATH = "./data/"
    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"
    
    # 模型參數
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 128
    NUM_CLASSES = 2  # 正面/負面情感
    
    # 訓練參數
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 10
    
    # 運行模式
    MODE = 'train'  # 'train' 或 'predict'
