"""
PIL Image -> torch.tensor
Resize
Compose
RandomCrop
"""

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")  # 创建实例

img_path = r"D:\python_code\Learn_Torch\hymenoptera_data\train\ants\196057951_63bf063b92.jpg"  # 图片的路径
img_PIL = Image.open(img_path)  # 用 Image 类的 open 方法打开图片
img_tensor_tool = transforms.ToTensor()  # 将 PIL 类转换成 torch.tensor 类（这里是先实例化，再传的参数）
img_tensor = img_tensor_tool(img_PIL)
writer.add_image("img_tensor", img_tensor)

img_resize_tool = transforms.Resize((512, 512))  # 指定长宽裁剪
img_resize_tool_2 = transforms.Resize(512)  # 将图片短边缩放至 512，长宽比保持不变裁剪
img_compose_tool = transforms.Compose([img_resize_tool, img_resize_tool_2])  # 创建 Compose 类实例
img_compose = img_compose_tool(img_tensor)  # 图片作为参数传入实例
writer.add_image("img_compose", img_compose)

img_crop_tool = transforms.RandomCrop(128)  # 创建 RandmCrop 类实例
for i in range(10):
    img_crop = img_crop_tool(img_tensor)  # 图片作为参数传入实例
    writer.add_image("img_crop", img_crop, i)

writer.close()  # 关闭实例
