import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from .net import NetSiamFC
from .transforms import TransformsSiamFC
from .dataset import Pair
from .config import cfg

def train_siamfc(seqs):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    # 数据集加载
    data_loader = DataLoader(
        Pair(seqs, transforms=TransformsSiamFC(
            cfg.exemplar_size, cfg.instance_size, cfg.context_amount)),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cuda,
        drop_last=True
    )
    # 模型
    net = NetSiamFC(cfg.score_scale).to(device)
    # 优化器设置
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=cfg.beg_lr,
        weight_decay=cfg.weight_decay,
        momentum=cfg.momentum
    )
    # 学习率衰减设置
    lr_scheduler = ExponentialLR(
        optimizer,
        np.power(cfg.end_lr / cfg.beg_lr, 1. / cfg.epoch_num)
    )
    # 最后的score_map大小固定，所以只需生成一次label
    labels, weight = _create_label_weight((cfg.score_size, cfg.score_size), device)
    # 训练过程记录
    if not os.path.exists(cfg.model_save_dir):
        os.mkdir(cfg.model_save_dir)
    if not os.path.exists(cfg.log_dir):
        os.mkdir(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    
    # 训练过程
    for epoch in range(cfg.epoch_num):
        print(f"EPOCH {epoch + 1}...")
        for i, batch in enumerate(tqdm(data_loader)):
            z = batch[0].to(device, non_blocking=True)
            x = batch[1].to(device, non_blocking=True)

            scores = net(z, x)
            loss = F.binary_cross_entropy_with_logits(
                scores, labels, weight, reduction='sum')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("training_loss",
                loss.item(), epoch * len(data_loader) + i)
        
        torch.save(net.state_dict(),
            os.path.join(cfg.model_save_dir, f"SiamFC_{epoch+1}.pth"))
        # 更新学习率
        lr_scheduler.step(epoch)

def _create_label_weight(size, device):
    h, w = size
    y = np.arange(h) - (h - 1) / 2
    x = np.arange(w) - (w - 1) / 2
    x, y = np.meshgrid(x, y)
    dist_map = np.sqrt(x**2 + y**2)
    labels = dist_map < cfg.pos_radius / cfg.stride
    labels = np.tile(labels, (cfg.batch_size, 1, 1, 1))
    labels = torch.FloatTensor(labels).to(device)

    pos_mask = labels == 1
    neg_mask = labels == 0
    weight = torch.zeros(labels.shape, device=device)
    weight[pos_mask] = 1. / torch.sum(pos_mask)
    weight[neg_mask] = 1. / torch.sum(neg_mask) * cfg.neg_weight
    weight /= torch.sum(weight)
    return labels, weight
