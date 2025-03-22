import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from LeNet5 import LeNet5

# 设备配置，检查是否有可用的 GPU，如果没有则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理：将图片转换为张量，并进行归一化处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将 PIL 图片转换为 PyTorch 张量
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化：使数据均值为 0，标准差为 1，提高训练稳定性
])

# 加载 MNIST 数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)  # 训练集
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)  # 测试集

# 使用 DataLoader 进行批量加载数据，提高训练效率
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 训练数据批量大小 64，打乱顺序
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 测试数据批量大小 64，不打乱顺序

# 加载 LeNet5 模型，并将其移动到设备（CPU 或 GPU）
model = LeNet5().to(device)

# 定义损失函数（交叉熵损失）和优化器（Adam）
criterion = nn.CrossEntropyLoss()  # 适用于多分类任务
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam 优化器，学习率 0.001

# 训练模型
def train(num_epochs=10):
    """训练 LeNet5 模型"""
    for epoch in range(num_epochs):  # 迭代多个训练轮次
        model.train()  # 设定模型为训练模式
        running_loss = 0.0  # 记录损失值
        for images, labels in train_loader:  # 遍历训练集的所有批次
            images, labels = images.to(device), labels.to(device)  # 将数据移动到 CPU 或 GPU

            optimizer.zero_grad()  # 清空梯度
            outputs = model(images)  # 前向传播，获取预测结果
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新模型参数

            running_loss += loss.item()  # 累加损失
        
        # 打印当前轮次的平均损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    print("Training complete.")  # 训练完成

# 评估模型
def test():
    """测试 LeNet5 模型的准确率"""
    model.eval()  # 设定模型为评估模式（不会更新梯度）
    correct = 0  # 统计正确预测的样本数
    total = 0  # 统计总样本数
    with torch.no_grad():  # 禁用梯度计算，加快推理速度，减少内存占用
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)  # 将数据移动到 CPU 或 GPU
            outputs = model(images)  # 获取预测结果
            _, predicted = torch.max(outputs, 1)  # 选取概率最高的类别作为最终预测
            total += labels.size(0)  # 统计样本总数
            correct += (predicted == labels).sum().item()  # 统计预测正确的样本数
    
    # 计算并打印测试准确率
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    train(10)  # 训练模型 10 轮
    test()  # 测试模型