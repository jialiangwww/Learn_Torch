"""
上节课我们的卷积层用的是
import torch.nn.functional as F
F。con2d()
这节课我们用的是
from torch.nn import Conv2d
Conv2d()
"""
import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import Conv2d
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

tudui = Tudui()

writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    print("The shape of inputs：", imgs.shape)
    print("The shape of outputs", outputs.shape)
    writer.add_images("inputs", imgs, step)
    outputs = torch.reshape(outputs, (-1, 3, 30, 30))  # 因为 TensorBoard 无法显示 channel 数为 6 的图片，
                                                       # 所以 channel：6->3，多余的压缩到 batch size 维度
    writer.add_images("outputs", outputs, step)
    step += 1
writer.close()
