import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # 進度條

# PyTorch 相關套件
import torch
import torch.nn as nn
import torch.nn.functional as F

import wget
import zipfile
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# 下載測試資料
# wget.download('https://github.com/TA-aiacademy/course_3.0/releases/download/DL/Data_part3.zip', 'Data_part3.zip')
# zip = zipfile.ZipFile('Data_part3.zip')
# zip.extractall()
# zip.close()

# 讀資料
print('讀取資料...')
train_df = pd.read_csv('./Data/News_train.csv')
test_df = pd.read_csv('./Data/News_test.csv')

print(train_df.head())

X_df = train_df.iloc[:, :-1].values
y_df = train_df.y_category.values

X_test = test_df.iloc[:, :-1].values
y_test = test_df.y_category.values

print(X_df, y_df)
print(X_test, y_test)


# 特徵縮放 Feature scaling
print('特徵縮放...')
sc = StandardScaler()
X_scale = sc.fit_transform(X_df, y_df)
X_test_scale = sc.transform(X_test)

X_train, X_valid, y_train, y_valid = train_test_split(X_scale, y_df,
                                                      test_size=0.2,
                                                      random_state=5566,
                                                      stratify=y_df)

print(f'X_train shape: {X_train.shape}')
print(f'X_valid shape: {X_valid.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_valid shape: {y_valid.shape}')

# 轉換為 PyTorch 的 Tensor
print('轉換為 PyTorch 的 Tensor...')
train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                          torch.tensor(y_train, dtype=torch.long))
valid_ds = torch.utils.data.TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                                          torch.tensor(y_valid, dtype=torch.long))

BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=BATCH_SIZE)

NUM_CLASS = 11

torch.manual_seed(5566)

def build_model(input_shape, num_class):
    model = nn.Sequential(
        nn.Linear(input_shape, 16),
        nn.Sigmoid(),
        nn.Linear(16, 16),
        nn.Sigmoid(),
        nn.Linear(16, num_class),
    )
    return model
# 建立模型
print('建立模型...')
model = build_model(X_train.shape[1], NUM_CLASS)
print(model)

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss() # 多元分類損失函數

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu') 
print(f'device: {device}')
model = model.to(device)

# 訓練模型
print('訓練模型...')
def train_epoch(model, optimizer, loss_fn, train_dataloader, val_dataloader):
    # 訓練一輪
    model.train()
    total_train_loss = 0
    total_train_correct = 0
    for x, y in tqdm(train_dataloader, leave=False):
        x, y = x.to(device), y.to(device) # 將資料移至GPU
        y_pred = model(x) # 計算預測值
        if type(loss_fn) != nn.CrossEntropyLoss:
            y_pred = F.softmax(y_pred, dim=1)
            y = F.one_hot(y, num_classes=NUM_CLASS).float() # one-hot encoding
        loss = loss_fn(y_pred, y) # 計算誤差
        optimizer.zero_grad() # 梯度歸零
        loss.backward() # 反向傳播計算梯度
        optimizer.step() # 更新模型參數

        total_train_loss += loss.item()
        if type(loss_fn) != nn.CrossEntropyLoss:
            y = y.argmax(dim=1).long()
        # 利用argmax計算最大值是第n個類別，與解答比對是否相同
        total_train_correct += ((y_pred.argmax(dim=1) == y).sum().item())

    # 驗證一輪
    model.eval()
    total_val_loss = 0
    total_val_correct = 0
    # 關閉梯度計算以加速
    with torch.no_grad():
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            if type(loss_fn) != nn.CrossEntropyLoss:
                y_pred = F.softmax(y_pred, dim=1)
                y = F.one_hot(y, num_classes=NUM_CLASS).float() # one-hot encoding
            loss = loss_fn(y_pred, y)
            total_val_loss += loss.item()
            # 利用argmax計算最大值是第n個類別，與解答比對是否相同
            if type(loss_fn) != nn.CrossEntropyLoss:
                y = y.argmax(dim=1).long()
            total_val_correct += ((y_pred.argmax(dim=1) == y).sum().item())

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_train_acc = total_train_correct / len(train_dataloader.dataset)
    avg_val_loss = total_val_loss / len(val_dataloader)
    avg_val_acc = total_val_correct / len(val_dataloader.dataset)

    return avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc

# 訓練模型
def run(model, optimizer, loss_fn, train_loader, valid_loader, verbose=1):
    train_loss_log = []
    val_loss_log = []
    train_acc_log = []
    val_acc_log = []
    for epoch in tqdm(range(20)):
        avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc = train_epoch(model, optimizer, loss_fn, train_loader, valid_loader)
        train_loss_log.append(avg_train_loss)
        val_loss_log.append(avg_val_loss)
        train_acc_log.append(avg_train_acc)
        val_acc_log.append(avg_val_acc)
        if verbose == 1:
            print(f'Epoch: {epoch}, Train Loss: {avg_train_loss:.3f}, Val Loss: {avg_val_loss:.3f} | Train Acc: {avg_train_acc:.3f}, Val Acc: {avg_val_acc:.3f}')
    return train_loss_log, train_acc_log, val_loss_log, val_acc_log

train_loss_log, train_acc_log, val_loss_log, val_acc_log = run(model, optimizer, loss_fn, train_loader, valid_loader)