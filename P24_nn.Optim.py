"""
优化器
注意损失函数和优化器怎么配合使用
"""

import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, CrossEntropyLoss

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, 1)

class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),  # 卷积层
            MaxPool2d(2),  # 最大池化层
            Conv2d(32, 32, 5, padding=2),  # 卷积层
            MaxPool2d(2),  # 最大池化层
            Conv2d(32, 64, 5, padding=2),  # 卷积层
            MaxPool2d(2),  # 最大池化层
            Flatten(),  # 展平层
            Linear(1024, 64),  # 线性层
            Linear(64, 10)  # 线性层
        )

    def forward(self, x):
        x = self.model1(x)
        return x


tudui = Tudui()  # 定义模型结构
loss_func = CrossEntropyLoss()  # 定义损失函数
optim = torch.optim.SGD(tudui.parameters(), lr=0.01)  # 定义优化器
for epoch in range(20) :
    running_loss = 0.0  # 每个 epoch 的损失
    for data in dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_func(outputs, targets)  # 计算损失
        optim.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optim.step()  # 更新参数
        running_loss += loss
    print(running_loss)