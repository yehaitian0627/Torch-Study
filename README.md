# TorchStudy
pytorch学习代码

1. 1_TorchTest:测试pytorch是否安装成功
2. 2_DatasetTest:载入图片并显示图像
3. 3_TensorboardTest:使用TensorBoard展示图片
4. 4_TransformsTest:使用Transforms修改图片
5. 5_Transforms&Dataset:下载并使用torchvision自带数据集
6. 6_DataloaderTest:使用dataloader按批次取数据
7. 7_ModuleTest:使用Module.nn生成一个神经网络的初始结构并重写其中的方法
8. 8_ConvolutionTest:简单的卷积模拟
9. 9_nnConv2dTest:在图片数据集上进行卷积的例子
10. 10_nnMaxPool2dTest:在图片数据集上进行池化的例子
11. 11_nnSigmoidTest:激活函数
12. 12_nnDropoutTest:Dropout测试
13. 13_nnLinearTest:线性层
14. 14_nnSequentialTest:Sequential创建一个简单的网络
![image](https://github.com/yehaitian0627/Torch-Study/assets/71301962/2c6b4159-b941-457f-93d8-b53119e6668c)

15. 15_nnLossTest:使用交叉熵损失函数将计算训练结果并反向传播
    - 如下图所示，在执行backward之前并没有进行反向传播，梯度为0：
    
    ![1709103317987](https://github.com/yehaitian0627/Torch-Study/assets/71301962/6e28d6ea-aa2d-4ce1-a7b5-2e61222ce306)
    
    - 如下图所示， 执行backward后梯度不为0，发生了反向传播：
      
    ![1709103484842](https://github.com/yehaitian0627/Torch-Study/assets/71301962/4376a832-ab35-4020-aca7-9a8b81cbb4cb)
    
16. 16_OptimTest:优化器进行优化，利用反向传播的梯度对参数进行优化
    - 如下图所示，经过20轮训练，通过优化器进行优化参数，损失函数明显降低：
      
    ![image](https://github.com/yehaitian0627/Torch-Study/assets/71301962/bedf4eb5-0535-487b-8633-2aa89d63c6fb)

17. 17_modelVGGTest:使用和修改VGG16网络
18. 18_SaveTest:两种存储现有网络的方式
    18_LoadTest:两种加载现有网络的方式
    - 两种方式需要配套使用，方式一的存储只能用方式一的读取
    - 官方推荐第二种读取方法
19. 19_WholeTest:整体测试（需要创建"net"文件夹以存储网络信息）
    - 神经网络存储在myNet1中
    测试结果用Tensorboard展示如下
    ![image](https://github.com/yehaitian0627/Torch-Study/assets/71301962/a506e6b7-d7f3-4dd8-a99c-e42a8be62160)
