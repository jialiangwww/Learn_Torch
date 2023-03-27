"""
需要调用 .cuda 的地方：
1. 网络模型
2. 损失函数
3. 数据（输入，标注）

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  定义训练的设备
x = x.to(device)  应用设备（上面说的那三个地方需要应用）
"""

import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Flatten, Linear
# from model import Tudui
import torch
from torch.utils.tensorboard import SummaryWriter

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
train_data = torchvision.datasets.CIFAR10("./CIFAR10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
validate_data = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor, download=True)

train_data_size = len(train_data)
validate_data_size = len(validate_data)
print(f"训练数据集长度为{train_data_size}")  # 50000
print(f"验证数据集长度为{validate_data_size}")  # 10000

# Dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(train_data, batch_size=64)


# model
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Sequential(
            Conv2d(3, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, 2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


tudui = Tudui()
tudui = tudui.to(device)

# loss
loss_func = CrossEntropyLoss()
loss_func = loss_func.to(device)

# optim
lr = 1e-2
optim = torch.optim.SGD(tudui.parameters(), lr=lr)

# train and validate
total_train_step = 0  # 训练次数
total_validate_step = 0  # 验证次数
epoch = 10  # 轮数
writer = SummaryWriter("logs")

for i in range(epoch):
    print(f"----------第 {i + 1} 次训练开始----------")

    # train
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)

        loss = loss_func(outputs, targets)  # 计算损失
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()  # 更新参数

        total_train_step += 1
        if total_train_step % 100 == 0:  # 每 100 次训练打印一次损失
            print(f"训练次数：{total_train_step}, Loss：{loss.item()}")
            writer.add_scalar("train_loss：", loss.item(), total_train_step)

    # validate
    tudui.eval()
    total_validate_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)

            loss = loss_func(outputs, targets)  # 损失
            total_validate_loss += loss.item()  # 总损失
            accuracy = (outputs.argmax(1) == targets).sum()  # 正确度
            total_accuracy += accuracy  # 总正确度

    print(f"整体验证集上的 Loss：{total_validate_loss}")
    print(f"整体验证集上的 Accuracy：{total_accuracy / validate_data_size}")
    writer.add_scalar("validate_loss", total_validate_loss, total_validate_step)
    writer.add_scalar("validate_accuracy", total_accuracy / validate_data_size, total_validate_step)
    total_validate_step += 1

    torch.save(tudui, f"tudui_{i}.pth")
    print("模型已保存")

writer.close()