import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Sequential_net(nn.Module):
    def __init__(self):
        super(Sequential_net, self).__init__()
        self.model1 = Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2, dilation=1),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2, padding=1),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2, stride=2, padding=0),
            Flatten(),
            Linear(in_features=1024, out_features=64),
            Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


net_test = Sequential_net()
print(net_test)
net_input = torch.ones([64, 3, 32, 32])
result_shape = net_test(net_input)
print(result_shape.shape)

# 查看网络结构
writer = SummaryWriter("logs/log_14")
writer.add_graph(net_test, net_input)
writer.close()