from torch.utils.data import Dataset
from PIL import Image
import os


class Mydata(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir  # 目录
        self.label_dir = label_dir  # 该数据集存放时，分别将两个类别存放在了两个文件夹之中，类别名称就是文件夹名称
        self.path = os.path.join(root_dir, label_dir)  # 拼接后是地址：目录+类别
        self.img_path = os.listdir(self.path)  # 生成数据集列表

    def __getitem__(self, idx):
        img_name = self.img_path[idx]  # 获取第idx个图片
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 拼接具体图像的地址：目录+类别+图像名称
        img = Image.open(img_item_path)  # 按图像地址打开图像
        label = self.label_dir  # 记录图像类别（文件夹名）
        return img, label  # 返回打开的图像和它的类别

    def __len__(self):
        return len(self.img_path)  # 返回数据集长度


root_dir = "dataset/train"  # 指定数据集目录
ants_label_dir = "ants"  # 文件夹名（类别）
bees_label_dir = "bees"
ants_dataset = Mydata(root_dir, ants_label_dir)  # 生成类对象调用方法，打开文件夹
bees_dataset = Mydata(root_dir, bees_label_dir)
dataset = ants_dataset + bees_dataset  # 对两个类对象进行拼接，也可以应用于两个数据集进行拼接的情况

print(len(dataset), len(ants_dataset), len(bees_dataset)) # 查看拼接后的数据集的总长度，以及其中两个不同类别的长度
idx = 124  # 指定输出第几张图片
img, label = dataset[idx]  # 取得返回值
print(label)  # 展示返回的图像
img.show()  # 输出返回的图像所属类别
