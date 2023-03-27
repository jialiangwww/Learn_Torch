"""
使用 torchvision 中的 datasets 读取 CIFAR10 数据集
"""

from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

dataset_compose_tool = transforms.Compose([  # 设置图片改变方式(全部转为 torch.tensor 类型)
    transforms.ToTensor()
])

train_set = datasets.CIFAR10(root="./CIFAR10", train=True, transform=dataset_compose_tool, download=True)  # 训练集
test_set = datasets.CIFAR10(root="./CIFAR10", train=False, transform=dataset_compose_tool, download=True)  # 测试集

# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()

writer = SummaryWriter("logs")  # 创建实例
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()  # 关闭实例
