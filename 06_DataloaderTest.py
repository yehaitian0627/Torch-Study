import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# transforms=对图像进行预处理
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True, transform=transforms.ToTensor())
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True, transform=transforms.ToTensor())

writer = SummaryWriter("logs/log_6")

# dataset=数据，batch_size=一次拿多少数据，shuffle=是否随机取数据，drop_last=是否舍弃最后剩余不成批的数据
test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=0, drop_last=False)

step = 0
for data in test_loader:
    # 每一批数据的图像存入 imgs，类别存入targets
    imgs, targets = data
    # 一次输出多个数据为一组，用 add_images而非 add_image
    writer.add_images("test_set", imgs, step)
    step += 1

writer.close()
