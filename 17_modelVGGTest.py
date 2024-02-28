import torchvision.models.vgg
from torch import nn

# pretrained=True 表示参数预训练过的网络
vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_false = torchvision.models.vgg16(pretrained=False)

print(vgg16_true)

# 在原有基础上添加一个线性层
vgg16_true.classifier.add_module("linear_7", nn.Linear(1000, 10))
print(vgg16_true)

# 改变原有最后一线性层的 out_features
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)