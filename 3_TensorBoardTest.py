from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")  # 将条目直接写入 log_dir 中的事件文件以供 TensorBoard 使用。
# tensorboard --logdir=logs  # 在终端中使用该语句可以打开 logs 文件


# 1.SummaryWriter.add_scalar(tag, scalar_value, global_step)
# tag (str): 图像标题
# scalar_value (float or string/blob-name): 值（y轴）
# global_step (int): 第几步（x轴）
for i in range(100):
    writer.add_scalar("y = x", 2 * i, i)  # 添加标量

# 2.SummaryWriter.add_image()
# tag (str): 标题
# img_tensor (torch.Tensor, numpy.ndarray, or string/blobname): 图像数据，注意图像格式
# global_step (int): Global step value to record
# walltime (float): Optional override default walltime (time.time()) seconds after epoch of event
# dataformats (str): Image data format specification of the form CHW, HWC, HW, WH, etc. 规定了图片的格式，可以通过 print(shape(img))获取
img_path = "dataset/train/ants/0013035.jpg"  # 目标图片的路径
img = Image.open(img_path)  # 打开路径获取图片
img_array = np.array(img)  # 将图片转化为 numpy 格式

writer.add_image("img", img_array, 1, dataformats="HWC")

writer.close()
