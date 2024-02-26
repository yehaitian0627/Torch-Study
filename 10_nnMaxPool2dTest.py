import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)
writer = SummaryWriter("logs/log_10")


class MaxPool2d_Net(nn.Module):
    def __init__(self):
        super(MaxPool2d_Net, self).__init__()
        # ceil_mode=True 在池化核伸展出图像后仍然进行池化
        self.maxPool = MaxPool2d(kernel_size=3, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        x = self.maxPool(x)
        return x


net = MaxPool2d_Net()

step = 0
for data in dataloader:
    imgs, targets = data
    output = net.maxPool(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", output, step)

    step += 1

writer.close()
