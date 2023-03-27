"""
Data
Dataloader
model
loss
optim
train（计算损失 -> 梯度清零 -> 反向传播 -> 更新参数）
validate
"""

import torchvision
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from model import Tudui
import torch
from torch.utils.tensorboard import SummaryWriter

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
tudui = Tudui()

# loss
loss_func = CrossEntropyLoss()

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