import torch
import torchvision
from torch import nn
from torch.nn import Dropout, Dropout2d, Dropout1d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 不舍弃最后不成批的数据就会导致 in_features不为 98304，就会报错
dataload = DataLoader(dataset=dataset, batch_size=32, shuffle=True, drop_last=True)


class Linear_net(nn.Module):
    def __init__(self):
        super(Linear_net, self).__init__()
        self.linear = Linear(98304, 10)

    def forward(self, input):
        output = self.linear(input)
        return output


net = Linear_net()

step = 0
for data in dataload:
    imgs, targets = data
    print(imgs.shape)
    # flatten:将 Tensor展开为一行
    imgs = torch.flatten(imgs)
    print(imgs.shape)
    output = net.linear(imgs)
    print(output)
    step += 1

