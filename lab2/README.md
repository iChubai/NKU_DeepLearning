# 深度学习实验2 - CNN网络结构实验

本实验实现了以下四种卷积神经网络结构，并在CIFAR-10/CIFAR-100数据集上进行了训练和验证：

1. 基础CNN网络
2. 微型ResNet网络
3. 微型DenseNet网络
4. 带SE结构的微型ResNet网络

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- matplotlib 3.4.0+
- numpy 1.19.0+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 文件结构

- `models.py`: 包含所有模型的定义
- `train.py`: 训练脚本
- `requirements.txt`: 项目依赖
- `README.md`: 项目说明文档

## 使用方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行训练脚本：
```bash
python train.py
```

## 实验结果

训练过程会自动保存以下文件：

- `best_BasicCNN.pth`: 基础CNN网络的最佳模型参数
- `best_MiniResNet.pth`: 微型ResNet网络的最佳模型参数
- `best_MiniDenseNet.pth`: 微型DenseNet网络的最佳模型参数
- `best_MiniSEResNet.pth`: 带SE结构的微型ResNet网络的最佳模型参数

同时会生成每个模型的训练曲线图：

- `BasicCNN_curves.png`: 基础CNN网络的训练曲线
- `MiniResNet_curves.png`: 微型ResNet网络的训练曲线
- `MiniDenseNet_curves.png`: 微型DenseNet网络的训练曲线
- `MiniSEResNet_curves.png`: 带SE结构的微型ResNet网络的训练曲线

## 网络结构说明

1. 基础CNN网络：
   - 3个卷积层
   - 2个全连接层
   - 使用ReLU激活函数和Dropout

2. 微型ResNet网络：
   - 1个初始卷积层
   - 3个残差块
   - 全局平均池化
   - 1个全连接层

3. 微型DenseNet网络：
   - 1个初始卷积层
   - 2个密集块
   - 1个过渡层
   - 全局平均池化
   - 1个全连接层

4. 带SE结构的微型ResNet网络：
   - 基于微型ResNet
   - 每个残差块增加SE注意力机制
   - SE模块使用全局平均池化和两个全连接层 