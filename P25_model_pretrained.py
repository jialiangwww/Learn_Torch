"""
调用已有的模型结构，并对其中部分结构进行修改
"""

import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)  # 没有被预训练过
vgg16_true = torchvision.models.vgg16(pretrained=True)  # 被预训练过

vgg16_false.classifier.add_module("add_linear", nn.Linear(1000, 10))  # 添加层
print(vgg16_false)
vgg16_true.classifier[6] = nn.Linear(4096, 10)  # 修改层。这里用的 vgg16_true 知识为了和上面的 vgg16_false 作区分，没别的用意
print(vgg16_true)
