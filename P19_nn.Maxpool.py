import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter

"""
input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1],
                      ], dtype=torch.float32)

print("The shape of input：", input.shape)
input = torch.reshape(input, (-1, 1, 5, 5))  # 看 MaxPool2d 文档后发现输入要求是 4 维的，所以这里要升维
"""

dataset = torchvision.datasets.CIFAR10("./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Tudui(nn.Module):

    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=True)  # 最大池化层的 stride 默认是 kernel_size

    def forward(self, input):
        output = self.maxpool(input)
        return output


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