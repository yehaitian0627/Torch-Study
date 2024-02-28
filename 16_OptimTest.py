import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataload = DataLoader(dataset=dataset, batch_size=32, shuffle=True)


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


# 初始化神经网络
net_test = Sequential_net()
# 交叉熵损失函数
loss = nn.CrossEntropyLoss()
# 设置优化器
optim = torch.optim.SGD(net_test.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0
    for data in dataload:
        imgs, targets = data
        # 神经网络结果
        output = net_test(imgs)
        # 获取损失函数值
        loss_result = loss(output, targets)
        # 将梯度清零
        optim.zero_grad()
        # 反向传播
        loss_result.backward()
        # 对参数进行调优
        optim.step()
        running_loss = running_loss + loss_result
    print(running_loss)

