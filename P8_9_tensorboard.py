"""
TensorBoard 的使用
运行结束后会生成 logs 文件，该文件下有一个很神奇的文件（不用打开它）
在 logs文件所在位置的终端下输入 tensorboard --logdir=logs，然后点击网址进入网页即可查看图片和图像在 TensorBoard 中的展现
"""
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")  # 创建实例

"""图片"""
img_path = r"D:\python_code\Learn_Torch\hymenoptera_data\train\ants\6240329_72c01e663e.jpg"  # 给定待读取图片的地址
img_PIL = Image.open(img_path)  # 以 PIL 形式读取图片
img_array = np.array(img_PIL)  # 将 PIL 形式的图片转为 numpy.ndarray 形式
writer.add_image("test_tensorboard", img_array, 1, dataformats='HWC')  # add_image() 方法只能接受 torch.tensor 或 numpy.ndarray

"""图像"""
for i in range(100):
    writer.add_scalar("y=2x", i * 2, i)

writer.close()  # 关闭实例
