import argparse
import torch
import torch.optim as optim
from torchvision import transforms
from data import load_mnist
from model import DigitRecognizer, train, test, save_model, load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def parse_arguments():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='手寫數字辨識系統')
    parser.add_argument('--mode', type=str, default='predict', 
                       choices=['train', 'predict'], help='運行模式: train 或 predict')
    parser.add_argument('--epochs', type=int, default=15, 
                       help='訓練的 epochs 數量')
    parser.add_argument('--image_path', type=str, 
                       help='預測用的圖片路徑')
    return parser.parse_args()

def get_device():
    """獲取可用設備 (GPU 或 CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    """訓練模型"""
    device = get_device()
    print(f"使用設備: {device}")
    
    # 加載數據
    train_loader, test_loader = load_mnist()
    
    # 初始化模型
    model = DigitRecognizer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練循環
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        print(f"Epoch {epoch}: 訓練損失={train_loss:.4f}, 測試損失={test_loss:.4f}, 準確率={accuracy:.2f}%")
    
    # 保存模型
    save_model(model)
    print("模型訓練完成並已保存")

def predict_image(image_path):
    """預測單張圖片"""
    device = get_device()
    print(f"使用設備: {device}")
    
    # 加載模型
    model = DigitRecognizer().to(device)
    model = load_model(model)
    model.eval()
    
    # 預處理圖片
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 進行預測
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
        probabilities = torch.exp(output).squeeze().tolist()
    
    # 顯示結果
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 設定支持中文的字體
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    plt.imshow(image, cmap='gray')
    plt.title(f"預測結果: {prediction}\n置信度: {probabilities[prediction]*100:.1f}%")
    plt.axis('off')
    plt.tight_layout()  # 自動調整佈局
    plt.show()

if __name__ == '__main__':
    args = parse_arguments()
    
    if args.mode == 'train':
        train_model()
    elif args.mode == 'predict':
        if not args.image_path:
            print("請提供圖片路徑 (--image_path)")
        else:
            predict_image(args.image_path)
