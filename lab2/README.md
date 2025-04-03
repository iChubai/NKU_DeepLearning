# ���ѧϰʵ��2 - CNN����ṹʵ��

��ʵ��ʵ�����������־��������ṹ������CIFAR-10/CIFAR-100���ݼ��Ͻ�����ѵ������֤��

1. ����CNN����
2. ΢��ResNet����
3. ΢��DenseNet����
4. ��SE�ṹ��΢��ResNet����

## ����Ҫ��

- Python 3.7+
- PyTorch 1.9.0+
- torchvision 0.10.0+
- matplotlib 3.4.0+
- numpy 1.19.0+

## ��װ����

```bash
pip install -r requirements.txt
```

## �ļ��ṹ

- `models.py`: ��������ģ�͵Ķ���
- `train.py`: ѵ���ű�
- `requirements.txt`: ��Ŀ����
- `README.md`: ��Ŀ˵���ĵ�

## ʹ�÷���

1. ��װ������
```bash
pip install -r requirements.txt
```

2. ����ѵ���ű���
```bash
python train.py
```

## ʵ����

ѵ�����̻��Զ����������ļ���

- `best_BasicCNN.pth`: ����CNN��������ģ�Ͳ���
- `best_MiniResNet.pth`: ΢��ResNet��������ģ�Ͳ���
- `best_MiniDenseNet.pth`: ΢��DenseNet��������ģ�Ͳ���
- `best_MiniSEResNet.pth`: ��SE�ṹ��΢��ResNet��������ģ�Ͳ���

ͬʱ������ÿ��ģ�͵�ѵ������ͼ��

- `BasicCNN_curves.png`: ����CNN�����ѵ������
- `MiniResNet_curves.png`: ΢��ResNet�����ѵ������
- `MiniDenseNet_curves.png`: ΢��DenseNet�����ѵ������
- `MiniSEResNet_curves.png`: ��SE�ṹ��΢��ResNet�����ѵ������

## ����ṹ˵��

1. ����CNN���磺
   - 3�������
   - 2��ȫ���Ӳ�
   - ʹ��ReLU�������Dropout

2. ΢��ResNet���磺
   - 1����ʼ�����
   - 3���в��
   - ȫ��ƽ���ػ�
   - 1��ȫ���Ӳ�

3. ΢��DenseNet���磺
   - 1����ʼ�����
   - 2���ܼ���
   - 1�����ɲ�
   - ȫ��ƽ���ػ�
   - 1��ȫ���Ӳ�

4. ��SE�ṹ��΢��ResNet���磺
   - ����΢��ResNet
   - ÿ���в������SEע��������
   - SEģ��ʹ��ȫ��ƽ���ػ�������ȫ���Ӳ� 