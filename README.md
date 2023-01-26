# SiamFC-pytorch

使用pytorch实现的一个比较简洁的SiamFC跟踪算法，在GOT-10k数据集上进行训练。融合了[GOT-10k toolkit](https://github.com/got-10k/toolkit)进行数据集读取和评价。

## Training

使用[tools/train.py](https://github.com/Stillwtm/SiamFC-pytorch/blob/main/tools/train.py)进行训练。

只需将[tools/train.py](https://github.com/Stillwtm/SiamFC-pytorch/blob/main/tools/train.py)中的`root_dir`参数更改为GOT-10k数据集的根目录，然后运行：

```bash
python -u tools/train.py
```

可以使用tensorboard跟踪训练过程。

## Evaluating

使用[tools/test.py](https://github.com/Stillwtm/SiamFC-pytorch/blob/main/tools/test.py)进行评价。

1. 首先将`net_path`参数改为训练好的模型文件位置；

2. 默认使用OTB100进行评价，将`data_dir`参数改为OTB100数据集的根目录，然后运行：

```bash
python tools/test.py
```

如果需要使用其他数据集进行评价，可以参考[GOT-10k toolkit](https://github.com/got-10k/toolkit)的使用，简单地替换成其他数据集。

## Configuration

训练和跟踪的所有参数配置全部在[siamfc/config.py](https://github.com/Stillwtm/SiamFC-pytorch/blob/main/siamfc/config.py)中，可以更改参数重新训练或者进行跟踪。

## Reference

1. [Bertinetto, Luca et al. “Fully-Convolutional Siamese Networks for Object Tracking.” *ECCV Workshops* (2016).](https://arxiv.org/pdf/1606.09549v3.pdf)

2. https://github.com/StrangerZhang/SiamFC-PyTorch

3. https://github.com/huanglianghua/siamfc-pytorch