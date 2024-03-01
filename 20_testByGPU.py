"""
    只有 神经网络、损失函数、数据集 有 GPU加速
    只要使用.cuda() 存在就是可以加速
"""
import time

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

# 创建设备
device = torch.device("cuda")

# 创建网络
my_net = myNet1.Net_test()
my_net = my_net.to(device)
# 创建损失函数
loss = nn.CrossEntropyLoss()
loss = loss.to(device)
# 创建优化器
lr = 0.008  # 优化器学习速率
optimizer = torch.optim.SGD(my_net.parameters(), lr=lr)

# 训练次数
train_step = 0
total_train_step = 0

# 测试次数
test_step = 0

# 训练轮数
epoch = 20

# 准确率
train_accuracy = 0
test_accuracy = 0


writer = SummaryWriter("logs/log_19")

for i in range(epoch):
    my_net.train()
    print(f"---------------------------------- epoch : {i+1} --------------------------------------")
    train_step = 0
    train_accuracy = 0
    step_accuracy = 0
    start_time = time.time()
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = my_net.sequential(imgs)
        loss_out = loss(outputs, targets)

        optimizer.zero_grad()
        loss_out.backward()
        optimizer.step()

        if (train_step + 1) % 100 == 0:
            print(f"Train_step : {train_step+1} | loss : {loss_out}")
            writer.add_scalar("TRAIN Loss", scalar_value=loss_out, global_step=total_train_step+1)
        train_step += 1
        total_train_step += 1

        # 计算准确率
        step_accuracy = (outputs.argmax(1) == targets).sum()
        train_accuracy = (train_accuracy + step_accuracy)
    train_accuracy_out = train_accuracy / train_dataset_size * 100
    print(f"epoch {i+1} train_accuracy : {'%.2f' % train_accuracy_out}%.")

    my_net.eval()
    print("------------------ Testing -------------------")
    # 测试数据不进行梯度更新
    test_step = 0
    total_loss = 0
    with torch.no_grad():
        step_accuracy = 0
        test_accuracy = 0
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = my_net.sequential(imgs)
            loss_out = loss(outputs, targets)

            if (test_step + 1) % 100 == 0:
                print(f"Test_step : {test_step+1} | loss : {loss_out}")

            test_step += 1
            total_loss = total_loss + loss_out
            step_accuracy = (outputs.argmax(1) == targets).sum()
            test_accuracy = step_accuracy + test_accuracy
    test_accuracy_out = test_accuracy / test_dataset_size * 100
    print(f"Epoch {i+1} total_loss : {total_loss}")
    writer.add_scalar("TEST Loss", scalar_value=total_loss, global_step=i+1)
    print(f"epoch {i+1} test_accuracy : {'%.2f' % test_accuracy_out}%.")

    torch.save(my_net, f"net/epoch{i+1}.pth")
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"epoch_time ： {epoch_time}")

writer.close()