"""
Dataset 类代码实战
介绍了我们怎么写自己的 Dataset 类，并需要覆写其中的__init()__、__getitem__()、__len__()
"""
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # hymenoptera_data/train
        self.label_dir = label_dir  # ants
        self.path = os.path.join(self.root_dir, self.label_dir)  # hymenoptera_data/train/ants
        self.img_path = os.listdir(self.path)  # 列出 hymenoptera_data/train/ants 路径下的所有图片名字

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取第 idx 张图片的名字
        img_item_path = os.path.join(self.path, img_name)  # 获取第 idx 张图片的路径
        img = Image.open(img_item_path)  # 使用 Image 中的 open 方法打开这张图片
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "hymenoptera_data/train"
ants_label_dir = "ants"
bees_label_dir = "bees"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)
train_dataset = ants_dataset + bees_dataset

img, label = ants_dataset[11]
img.show()  # PIL 类型的图片可以通过 show() 方法予以展现
