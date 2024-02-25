import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# 下载数据集，root=路径，train=是否为训练集，download=是否自动下载
train_set = torchvision.datasets.CIFAR10(root="./dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./dataset", train=False, download=True)

writer = SummaryWriter("logs/log_5")
trans_tensor_tool = transforms.ToTensor()
trans_resize_tool = transforms.Resize([32, 32])

for i in range(0, 10):
    img, target = train_set[i]
    img_tensor = trans_tensor_tool(img)
    img_resize = trans_resize_tool(img_tensor)
    writer.add_image("train_test", img_resize, i)

writer.close()