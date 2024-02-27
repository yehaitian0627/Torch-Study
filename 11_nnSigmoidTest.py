import torch
import torchvision
from torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

writer = SummaryWriter("logs/log_11")


class Sigmoid_Net(nn.Module):
    def __init__(self):
        super(Sigmoid_Net, self).__init__()
        self.sigmoid = Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output

sigmoid_net = Sigmoid_Net()

step = 0
for data in dataloader:
    imgs, targets = data
    net_oput = sigmoid_net(imgs)
    writer.add_images("input", imgs, step)
    writer.add_images("output", net_oput, step)
    step += 1

writer.close()

