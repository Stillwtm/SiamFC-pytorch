import cv2
import numpy as np
import bisect
from torch.utils.data import Dataset

class Pair(Dataset):
    def __init__(self, seqs, transforms=None, frame_range=100):
        self.seqs = seqs
        self.transforms = transforms
        self.frame_range = frame_range
        self.return_meta = getattr(seqs, 'return_meta', False)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        """从第index个seq中随机选取一对图片"""
        if self.return_meta:
            img_files, annos, meta = self.seqs[index]
        else:
            img_files, annos = self.seqs[index]
        
        indices = self._filter(cv2.imread(img_files[0]), annos)
        # 如果这个seq中的图全被筛掉了就换一个seq
        if len(indices) < 2:
            index = np.random.choice(len(self.seqs))
            return self.__getitem__(index)
        # 从seq中采样一对图片
        z_idx, x_idx = self._sample(indices, len(self.seqs))
        
        z = cv2.cvtColor(cv2.imread(img_files[z_idx]), cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(cv2.imread(img_files[x_idx]), cv2.COLOR_BGR2RGB)
        box_z = annos[z_idx]
        box_x = annos[x_idx]
        if self.transforms is not None:
            z, x = self.transforms(z, x, box_z, box_x)
        return z, x

    def _sample(self, indices, n):
        z_idx = np.random.choice(indices, replace=False)
        min_idx = max(0, z_idx - self.frame_range)
        max_idx = min(n, z_idx + self.frame_range)
        min_idx = bisect.bisect_left(indices, min_idx)
        max_idx = bisect.bisect_left(indices, max_idx)
        x_idx = indices[np.random.randint(min_idx, max_idx)]
        return z_idx, x_idx

    def _filter(self, img0, annos):
        sz = np.array(img0.shape[1::-1])[None, :]  # 整张图片尺寸
        areas = annos[:, 2] * annos[:, 3]  # 标注框面积
        # 筛选规则
        r1 = areas >= 20
        r2 = np.all(annos[:, 2:] / sz >= 0.01, axis=1)
        r3 = np.all(annos[:, 2:] / sz <= 0.5, axis=1)  # 标注框的面积在全图中占比合适
        r4 = (annos[:, 2] / (annos[:, 3] + 1e-10) >= 0.25)
        r5 = (annos[:, 2] / (annos[:, 3] + 1e-10) <= 4)  # 长宽比合适
        mask = np.logical_and.reduce([r1, r2, r3, r4, r5])
        return np.nonzero(mask)[0]