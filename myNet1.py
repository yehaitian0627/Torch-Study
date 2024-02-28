import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear


class Net_test(nn.Module):
    def __init__(self):
        super(Net_test, self).__init__()
        self.sequential = Sequential(
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
        x = self.sequential(x)
        return x


if __name__ == '__main__':
    net = Net_test()
    inputs = torch.ones((32, 3, 32, 32))
    outputs = net.sequential(inputs)
    print(outputs.shape)
