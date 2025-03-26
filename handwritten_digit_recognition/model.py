import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DigitRecognizer(nn.Module):
    """CNN 模型用於手寫數字識別"""
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input_tensor):
        """前向傳播"""
        # 卷積層
        x = F.relu(self.conv1(input_tensor))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        
        # 全連接層
        x = x.view(-1, 128*3*3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    """訓練模型"""
    model.train()
    total_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        batch_loss = F.nll_loss(predictions, labels)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    return total_loss / len(train_loader)

def test(model, device, test_loader):
    """評估模型"""
    model.eval()
    test_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            test_loss += F.nll_loss(predictions, labels, reduction='sum').item()
            predicted_labels = predictions.argmax(dim=1, keepdim=True)
            correct_predictions += predicted_labels.eq(labels.view_as(predicted_labels)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct_predictions / len(test_loader.dataset)
    return test_loss, accuracy

def save_model(model, path='model.pth'):
    """保存模型"""
    torch.save(model.state_dict(), path)

def load_model(model, path='model.pth'):
    """加載模型"""
    model.load_state_dict(torch.load(path))
    return model
