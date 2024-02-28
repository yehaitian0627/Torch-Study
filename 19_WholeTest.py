import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import myNet1


train_dataset = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=True)

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)
print(train_dataset)
print(test_dataset)

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

# 创建网络
my_net = myNet1.Net_test()
# 创建损失函数
loss = nn.CrossEntropyLoss()
# 创建优化器
lr = 0.01  # 优化器学习速率
optimizer = torch.optim.SGD(my_net.parameters(), lr=lr)

# 训练次数
train_step = 0
total_train_step = 0

# 测试次数
test_step = 0

# 训练轮数
epoch = 12

writer = SummaryWriter("logs/log_19")

for i in range(epoch):
    print(f"---------------------------------- epoch : {i+1} --------------------------------------")
    train_step = 0
    for data in train_loader:
        imgs, targets = data
        outputs = my_net.sequential(imgs)
        loss_out = loss(outputs, targets)

        optimizer.zero_grad()
        loss_out.backward()
        optimizer.step()

        if (train_step + 1) % 100 == 0:
            print(f"Train_step : {train_step+1} | loss : {loss_out}")
            writer.add_scalar("train_loss", scalar_value=loss_out, global_step=total_train_step+1)
        train_step += 1
        total_train_step += 1

    print(f"------------------ Testing -------------------")
    # 测试数据不进行梯度更新
    test_step = 0
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            outputs = my_net.sequential(imgs)
            loss_out = loss(outputs, targets)

            if (test_step + 1) % 100 == 0:
                print(f"Test_step : {test_step+1} | loss : {loss_out}")

            test_step += 1
            total_loss = total_loss + loss_out
    print(f"Epoch {i+1} total_loss : {total_loss}")
    writer.add_scalar("Test loss", scalar_value=total_loss, global_step=i+1)

    torch.save(my_net, f"net/epoch{i+1}.pth")

writer.close()