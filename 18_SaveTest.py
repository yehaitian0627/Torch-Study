import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=True)

# 存储方式一
torch.save(vgg16, "nn/vgg16True1.pth")

# 存储方式二
torch.save(vgg16.state_dict(), "nn/vgg16True2.pth")