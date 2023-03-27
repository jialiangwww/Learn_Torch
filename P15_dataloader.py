"""
DataLoader 的使用，注意各个参数的含义
batch_size：批量大小
shuffle：是否随机打乱数据顺序
num_workers：工作线程数
drop_last：是否丢弃由于除批量大小不尽而余下来的数据

另外要注意 SummaryWriter 实例的 add_images 和 add_image 的区别
"""
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter("logs")
for epoch in range(2):
    step = 0
    for data in test_dataloader:
        imgs, targets = data
        writer.add_images("Epoch；{}".format(epoch), imgs, step)  # 注意这里是 add_images，因为添加进去的是一批图片而不是一张
        step += 1
writer.close()
