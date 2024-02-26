import torch
# torch.nn.functional和 torch.nn的区别在于：前者有底层实现，后者更注重使用（封装）
import torch.nn.functional as F

# 简单的卷积模拟

# 输入
conv_input = torch.tensor([[1, 2, 4, 5, 7],
                      [6, 3, 6, 1, 7],
                      [2, 5, 7, 1, 4],
                      [3, 4, 6, 2, 7],
                      [2, 6, 7, 2, 6],])

# 卷积核
kernel = torch.tensor([[1, 4, 6],
                       [4, 6, 2],
                       [3, 7, 8]])

# 将输入和卷积核变成 conv2d方法需要的形式
conv_input = torch.reshape(conv_input, [1, 1, 5, 5])
kernel = torch.reshape(kernel, [1, 1, 3, 3])

# 进行卷积 参数为：输入、卷积核、步长（卷积核每次的移动距离、扩充边距（默认为 0））
conv_output = F.conv2d(conv_input, kernel, stride=2, padding=1)

print(conv_output)
