import torch

# 验证torch是否安装成功

# 查看torch版本
print(torch.version)

# 查看cuda版本
print(torch.version.cuda)

# GPU是否可用
print(torch.cuda.is_available())

# 返回gpu数量
print(torch.cuda.device_count())

# 返回gpu名字，设备索引默认从0开始
print(torch.cuda.get_device_name(0))

# 例子
print(torch.rand(3, 3).cuda())

# 看Python方法的工具
print(dir(torch))  # open package
print(dir('AVG'))
print(help('__dir__'))  # open function


