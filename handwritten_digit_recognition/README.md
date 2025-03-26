# 手寫數字辨識系統

create by deepseek/deepseek-chat-v3-0324:free

## user prompt
```
設計並實現一個基於 PyTorch 的手寫數字辨識系統，可以做即時預測結果 
## 主要需求
- 我要生成一個 python 專案，
- 根目錄在 genAI-lab 下再開一層目錄
- 使用 PyTorch 框架構建 CNN 模型
- 要支持數字 (0-9) 辨識
- 提供實時預測結果

## 請提供以下代碼實現
1. 模型設計 (model.py)
* 適合 MNIST 的 CNN 架構
* 訓練和評估函數
2. 數據處理 (data.py)
* MNIST 數據加載與預處理
3. 主程序 (main.py)
* 可以設定主要的參數
* 提供訓練與推理功能
* 可以顯示相關的數據
* 使用者可以提供一張照片或多張照片來預測結果 請使用清晰的代碼注釋與適當的錯誤處理，並確保所有模塊之間能 夠無縫協作。
```

## system prompt
```
Run unit tests at the end.
Check if there are any undeclared variable names or unimported
packages.
Please first plan the order of all files to be written and function references properly.
For existing files, only check if there will be errors when executed and make corrections, avoid circular repetitive editing.
Speak in Traditional Chinese.
Don't reply with code content in the conversation, directly edit it to the corresponding file or provide relevant suggestions.
use meaningful words define parameters and arguments.
```

基於 PyTorch 的 CNN 手寫數字辨識系統，支援訓練和即時預測功能。

## 功能特點
- 使用 CNN 卷積神經網路識別手寫數字 (0-9)
- 自動檢測並使用 GPU 加速運算
- 支援訓練新模型或載入已有模型
- 提供即時預測功能
- 顯示預測結果與置信度

## 安裝需求

```bash
pip install -r requirements.txt
```

## 使用方法

### 訓練模型
```bash
python main.py --mode train --epochs 5
```
- `--epochs`: 設定訓練週期數 (預設為5)

### 使用模型預測圖片
```bash
python main.py --mode predict --image_path 圖片路徑
```
- `--image_path`: 指定要預測的圖片路徑

## 模型架構
- 2 個卷積層 (Conv2d)
- 2 個最大池化層 (MaxPool2d)
- 2 個全連接層 (Linear)
- Dropout 正則化 (比例0.25)

## 注意事項
1. 首次執行會自動下載 MNIST 數據集
2. 訓練完成的模型會自動保存為 model.pth
3. 預測圖片建議使用 28x28 像素的灰階圖片
4. 系統會自動選擇使用 GPU 或 CPU 運算
