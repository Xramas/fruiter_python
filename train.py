import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from model import Net # 假设你已经更新了 model.py
import json
import os
import argparse # 引入 argparse

# 1. 设置随机种子
torch.manual_seed(42)

# 2. Argparse 用于超参数管理
parser = argparse.ArgumentParser(description='Train a fruit classifier.')
parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# transforms 用于数据预处理和增强
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# 3. 划分训练集和验证集
full_dataset = torchvision.datasets.ImageFolder(root="data", transform=transform)
val_size = int(len(full_dataset) * 0.2) # 20% 作为验证集
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

class_names = full_dataset.classes
num_classes = len(class_names)

model = Net(num_classes=num_classes).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train():
    best_val_acc = 0.0 # 记录最佳验证准确率

    for epoch in range(args.epochs):
        # --- 训练循环 ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- 验证循环 ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")

        # 4. 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print(f"新最佳模型已保存，准确率: {val_acc:.2f}%")

    # 保存类别文件
    with open("classes.json", "w") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=4)
    print("训练完成。")

if __name__ == '__main__':
    train()