import torch
import torchvision.transforms
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset=dataset, batch_size=64, num_workers=0)


class conv2dNet(nn.Module):
    def __init__(self):
        super(conv2dNet, self).__init__()
        # in_channels 和 out_channels 是图像输入和输出的通道数（灰度图为 1、图片为 3、高光谱图像的通道数成百上千）
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


conv2dnet_test = conv2dNet()

writer = SummaryWriter("logs/log_9")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)

    net_output = conv2dnet_test(imgs)
    # 网络规定，输出图像为 6通道，想以 3通道的方式输出，就要把 batch_size变成两倍，设置 -1表示自动决定大小
    output = torch.reshape(net_output, [-1, 3, 30, 30])
    writer.add_images("output", output, step)

    step += 1

writer.close()
