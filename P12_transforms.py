"""
数据标准化 —— transforms.Normalize()

功能：逐 channel 的对图像进行标准化（均值变为 0，标准差变为 1），可以加快模型的收敛
output = (input - mean) / std
mean:各通道的均值
std：各通道的标准差
inplace：是否原地操作

"""

from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from torchvision import transforms

writer = SummaryWriter("logs")  # 创建实例

img_path = r"D:\python_code\Learn_Torch\hymenoptera_data\train\bees\21399619_3e61e5bb6f.jpg"  # 图片的路径
img_PIL = Image.open(img_path)  # 以 PIL 形式打开图片
img_tensor_tool = transforms.ToTensor()  # 将 PIL 类转换成 torch.tensor 类（这里是先实例化，再传的参数）
img_tensor = img_tensor_tool(img_PIL)

print("img_tensor[0][0][0]：", img_tensor[0][0][0])
img_norm_tool = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 实例化一个对象，即调用 Normalize 类的 __init__()方法 <-> 用工具箱中的工具做出一个属于自己的工具
img_norm = img_norm_tool(img_tensor)  # 给对象传入参数，即调用对象的 forward 方法 <-> 用做出来的工具去干活
print("img_norm[0][0][0]：", img_norm[0][0][0])
writer.add_image("Normalize", img_norm)  # 写入 TensorBoard

writer.close()
