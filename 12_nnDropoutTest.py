import torch
import torchvision
from torch import nn
from torch.nn import Dropout, Dropout2d, Dropout1d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(), download=True)
dataload = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

writer = SummaryWriter("logs/log_12")


class Dropout_net(nn.Module):
    def __init__(self):
        super(Dropout_net, self).__init__()
        self.dropout1 = Dropout()
        self.dropout2 = Dropout2d()

    def forward(self, input):
        output1 = self.dropout1(input)
        output2 = self.dropout2(input)
        return output1, output2


net = Dropout_net()

step = 0
for data in dataload:
    imgs, targets = data
    # print(imgs.shape()) [3, 32, 32] -> [-1, 3, 32, 32]
    imgs = torch.reshape(imgs, [-1, 3, 32, 32])
    writer.add_images("input", imgs, step)
    output1 = net.dropout1(imgs)
    output2 = net.dropout2(imgs)
    writer.add_images("dropout1", output1, step)
    writer.add_images("dropout2", output2, step)
    step += 1

writer.close()
