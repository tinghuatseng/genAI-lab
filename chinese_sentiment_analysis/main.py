from train import Trainer
from predict import Predictor
from config import Config

def main():
    config = Config()
    
    # 訓練模式
    if config.MODE == 'train':
        trainer = Trainer()
        trainer.train()
    # 預測模式
    elif config.MODE == 'predict':
        predictor = Predictor()
        while True:
            text = input("請輸入要分析的文本(輸入q退出): ")
            if text.lower() == 'q':
                break
            result = predictor.predict(text)
            print(f"情感分析結果: {result}")

if __name__ == "__main__":
    main()
