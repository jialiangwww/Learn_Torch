"""
用 transforms 里面的 ToTensor() 方法将 PIL 类型变成 torch.Tensor 类型
"""
from PIL import Image
from torchvision import transforms

img_path = r"D:\python_code\Learn_Torch\hymenoptera_data\train\ants\82852639_52b7f7f5e3.jpg"  # 图片的相对路径
img = Image.open(img_path)  # 使用 Image 类的 open 方法打开相对路径指定的图片

img_tensor_tool = transforms.ToTensor()  # 实例化一个对象，即调用 ToTensor 类的 __init__()方法 <-> 用工具箱中的工具做出一个属于自己的工具
tensor_img = img_tensor_tool(img)  # 给对象传入参数，即调用对象的__call__()方法 <-> 用做出来的工具去干活
"""
上面这里为什么不写 tensor_img = transforms.ToTensor(img) 呢，是因为观察 ToTensor 源码后发现，其初始化方法是
__init__(self)，只有一个参数，所以要用“先实例化对象，再给对象传入参数”的写法
 """

print(tensor_img)  # 打印张量
print(tensor_img.shape)  # 打印张良形状
