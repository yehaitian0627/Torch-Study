import torch
from torch import nn


class Net_test(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


net_test = Net_test()
tensor1 = torch.tensor(1.0)
print(net_test(tensor1))
