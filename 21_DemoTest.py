import torch
import torchvision
from PIL import Image
from torch import nn

import myNet1

# 1.dog图片测试
img_path = "./dataset/demo/dog3.jpg"
img1 = Image.open(img_path)

transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                             torchvision.transforms.ToTensor()]
                                            )

img1 = transforms(img1)
print(img1.shape)

net = myNet1.Net_test()
# epoch20.pth由 GPU训练得到的，需要将其部署在 CPU上才能正常使用
model = torch.load("net/epoch20.pth", map_location=torch.device("cpu"))
print(model)

img1 = torch.reshape(img1, (1, 3, 32, 32))
print(img1.shape)
model.eval()
with torch.no_grad():
    output = model(img1)

"""
    airplane = 0
    automobile = 1
    bird = 2
    cat = 3
    deer = 4
    dog = 5
    frog = 6
    horse = 7
    ship = 8
    truck = 9
"""
print(output.argmax(1))
