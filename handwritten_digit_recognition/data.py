import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def load_mnist(batch_size=128):
    """加載 MNIST 數據集並返回數據加載器"""
    # 定義數據轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加載訓練集和測試集
    train_set = datasets.MNIST(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    test_set = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # 創建數據加載器
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

if __name__ == '__main__':
    # 測試數據加載
    train_loader, test_loader = load_mnist()
    print(f"訓練集批次數: {len(train_loader)}")
    print(f"測試集批次數: {len(test_loader)}")
