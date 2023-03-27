import torchvision
import torch
from torch import nn

vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式 1；模型结构 + 模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式 2:模型参数（官方推荐）
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


# 陷阱(其实现在已经不是陷阱了，自己定义的模型也可以直接 load 了)
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


tudui = Tudui()
torch.save(tudui, "tudui_method1.pth")