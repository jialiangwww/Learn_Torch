"""
nn.Linear（）是用于设置网络中的全连接层的，
原型就是初级数学里学到的线性函数：y=kx+b，
不过在深度学习中，变量都是多维张量，乘法就是矩阵乘法，加法就是矩阵加法，因此nn.Linear()运行的真正的计算就是：
output = weight @ input + bias
"""

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Linear

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, 64, drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(196608, 10)

    def forward(self, inputs):
        outputs = self.linear(inputs)
        return outputs


tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    print("The shape of imgs：", imgs.shape)
    imgs_flatten = torch.flatten(imgs)  # imgs = torch.reshape(1, 1, 1, -1)
    print("The shape of imgs_flatten：", imgs_flatten.shape)
    outputs = tudui(imgs_flatten)
    print("The shape of outputs：", outputs.shape)
