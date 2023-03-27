"""
损失函数的作用：
1. 计算实际输出与目标之间的差距
2. 为我们更新输出提供一定的依据（反向传播）
"""

import torchvision
from torch.utils.data import DataLoader
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch import nn

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


tudui = Tudui()
loss_crossentropy = nn.CrossEntropyLoss()
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)  # 注意，这里的 outputs 不是分类概率，而是得分
    result = loss_crossentropy(outputs, targets)
    # result.backward()
    print(result)
