"""
可以使用 nn.Sequential 集成多个网络层
add_graph() 方法可以将模型结构打印出来
"""

import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.tensorboard import SummaryWriter


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


tudui = Tudui()
input = torch.ones(64, 3, 32, 32)
output = tudui(input)
print("The model structure is：\n", tudui)
print("The shape of input is：\n", input.shape)
print("The shape of output is：\n", output.shape)

writer = SummaryWriter("logs")
writer.add_graph(tudui, input)  # 打印模型结构用 add_graph()
writer.close()
