import torch
import torchvision

# 读取方式 1：模型结构 + 模型参数
model = torch.load("vgg16_method1.pth")
# print(model)

# 读取方式 2,：加载模型
# model = torch.load("vgg16_method2.pth")
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# print(vgg16)

# 陷阱(其实现在已经不是陷阱了，自己定义的模型也可以直接 load 了，下面按照老版本的讲)
model = torch.load("tudui_method1.pth")
print(model)  # FileNotFoundError: [Errno 2] No such file or directory: 'tudui_method1.pth'
"""
解决方法1：将 Tudui 这个类的定义代码粘过来
解决方法2： from model_save import Tudui
"""