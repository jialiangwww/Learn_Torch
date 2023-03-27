"""
利用训练好的模型对图片进行分类
"""
import torch
import torchvision
from PIL import Image
from torch import nn
from model import Tudui

image_path = "./imgs/truck.jpg"
image = Image.open(image_path)
# print(image)  # 可以看到是以 PIL 形式打开的图片
# image = image.convert('RGB')  因为 .jpg文件 本身就是 RGB 通道，所以这里不用转换（土堆的图片是 png 四个通道的，所以要转换）
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),torchvision.transforms.ToTensor()])
image = transform(image)
# print(image.shape)

model = torch.load("tudui_29.pth", map_location=torch.device('cpu'))  # 在 cuda 上训练得到的模型文件要想在 cpu 设备上读取，就得加上 map_location=torch.device('cpu')
image = torch.reshape(image, (1, 3, 32, 32))  # 输入 Tudui 的要是四维的
model.eval()  # 这个加上是一个好习惯
with torch.no_grad():  # 这个加上是一个好习惯
    output = model(image)
print(output)
print(output.argmax(1))
