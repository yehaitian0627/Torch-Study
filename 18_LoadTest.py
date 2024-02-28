import torch
import torchvision

# 加载方式一
model1 = torch.load("nn/vgg16True1.pth")
print(f"way1 : {model1}")

# 加载方式二
vgg16_True = torchvision.models.vgg16(pretrained=True)
model2 = vgg16_True.load_state_dict(torch.load("nn/vgg16True2.pth"))
print(f"way2 : {model2}\n {vgg16_True}")
