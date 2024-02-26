from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import cv2

# 打开图片
img_path = "dataset/train/ants/0013035.jpg"
# 用 Image获取图片后，得到一个 PIL.JpegImagePlugin.JpegImageFile类型的图片数据
img1 = Image.open(img_path)
# 用 opencv获取图片后，得到一个 numpy.ndarray类型的图片数据
img2 = cv2.imread(img_path)
print(f"The type of img_cv is {type(img2)} while the img_IMG's is {type(img1)}")
# transforms 工具箱，将图片输入后通过工具得到一个规范化的输出
# ToTensor需要 PIL Image或者 numpy.ndarray类型作为参数

# ---------------------------------------------------------------------------------------------
# 1.ToTensor
# 将数据转化为 Tensor类型

# a.transforms 的使用
trans_tensor_tool = transforms.ToTensor()  # 从工具箱中创建 tensor_trans_tool 工具
tensor_img1 = trans_tensor_tool(img1)  # 使用工具将 img 转换为 Tensor 的格式
tensor_img2 = trans_tensor_tool(img2)
print(tensor_img1)

# b.使用 Tensor 的原因
# Tensor中拥有反向神经网络理论基础的一些参数

# ctrl+alt+v 快速生成返回值
writer = SummaryWriter("logs/log_4")

writer.add_image("Tensor_img", tensor_img2)

# 没有 close写不进去
# writer.close()

# ---------------------------------------------------------------------------------------------
# 2.Normalize
# 将一个 Tensor类型的数据进行归一化，需要提供平均值(mean)和标准差(std)

trans_norm_tool = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm_tool(tensor_img1)

# 显示归一化后的图片
writer.add_image("Normalize", img_norm)

# ---------------------------------------------------------------------------------------------
# 3.Resize
# 剪裁
print(img1.size)
trans_resize_tool = transforms.Resize((512, 512))
img_resize = trans_tensor_tool(trans_resize_tool(img1))  # 想使用 tensorboard显示必须转化为 Tensor数据

# 显示裁剪后的图片
writer.add_image("Resize", img_resize)

# ---------------------------------------------------------------------------------------------
# 4.Compose
# 组合，输入一个序列，最终得到一个结果
trans_compose_tool = transforms.Resize(512)
# PIL -> PIL -> Tensor
trans_compose = transforms.Compose([trans_compose_tool, trans_tensor_tool])
img_compose = trans_compose(img1)
writer.add_image("Resize", img_compose, 1)

# ---------------------------------------------------------------------------------------------
# 5.RandomCrop
# 随机裁剪
trans_random_tool = transforms.RandomCrop(128)  # 按 128 * 128 裁剪，或者可以按照 [128 * 256] 这种形式限制其长和宽
trans_random = trans_random_tool(img1)
for i in range(0, 5):
    trans_random = trans_random_tool(img1)
    random_result = trans_tensor_tool(trans_random)
    writer.add_image("Random", random_result, i)

writer.close()
