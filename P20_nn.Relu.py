"""
激活函数有 relu, sigmoid 等
"""

import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, inputs):
        outputs = self.sigmoid(inputs)
        return outputs


tudui = Tudui()
writer = SummaryWriter("logs")
step = 0
for data in dataloader:
    imgs, targets = data
    outputs = tudui(imgs)
    writer.add_images("inputs", imgs, step)
    writer.add_images("outputs", outputs, step)
    step += 1
writer.close()