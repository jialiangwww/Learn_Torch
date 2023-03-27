"""
在 P8_9 中，我们将 PIL 类型转换成 numpy.ndarray 类型，在本节我们将 PIL 类型转换成 torch.tensor 类型
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")  # 创建实例

img_path = r"D:\python_code\Learn_Torch\hymenoptera_data\train\ants\149244013_c529578289.jpg"  # 图片的相对路径
img_PIL = Image.open(img_path)  # 用 Image 类的 open 方法打开图片
img_tensor_tool = transforms.ToTensor()  # 将 PIL 类转换成 torch.tensor 类（这里是先实例化，再传的参数）
img_tensor = img_tensor_tool(img_PIL)
writer.add_image("test_transforms", img_tensor, 1)

writer.close()  # 关闭实例
