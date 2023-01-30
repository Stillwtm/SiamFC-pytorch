import torch
import numpy as np
import cv2
import time
from PIL import Image
from got10k.trackers import Tracker
from got10k.utils.viz import show_frame
from .net import NetSiamFC
from .transforms import crop_resize_box, crop_resize_center, to_tensor
from .config import cfg


def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # convert box to corners (0-indexed)
    size = round(size)
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # pad image if necessary
    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size),
                       interpolation=interp)

    return patch

class TrackerSiamFC(Tracker):
    def __init__(self, model_path):
        super(TrackerSiamFC, self).__init__('SiamFC', True)

        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.net = NetSiamFC(score_scale=1.0)  # 推理时不需要再缩放
        model_dict = torch.load(model_path)
        if 'model' in model_dict:
            model_dict = model_dict['model']
        self.net.load_state_dict(model_dict)
        self.net = self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def init(self, img, box):
        # 二维余弦窗口
        self.cos_window = np.outer(
            np.hanning(cfg.upscale_size),
            np.hanning(cfg.upscale_size)
        )
        self.cos_window /= self.cos_window.sum()
        # 搜索尺寸
        scale_exps = np.arange(cfg.scale_num) - (cfg.scale_num - 1) / 2
        self.scale_factors = cfg.scale_step**scale_exps

        self.center = box[:2] + box[2:] / 2
        self.target_size = box[2:]
        z, self.z_size = crop_resize_box(img, box, cfg.context_amount, 
            cfg.exemplar_size, cfg.exemplar_size)
        
        self.x_size = self.z_size / cfg.exemplar_size * cfg.instance_size
        self.kernel = self.net.backbone(
            to_tensor(z).to(self.device).unsqueeze(0))
        self.kernel = self.kernel.repeat(cfg.scale_num, 1, 1, 1)

    @torch.no_grad()
    def update(self, img):
        # 多个尺寸的搜索图像合并成一个tensor
        xs = [crop_resize_center(
            img, self.center, 
            (self.x_size * s, self.x_size * s),
            (cfg.instance_size, cfg.instance_size))
            for s in self.scale_factors]

        xs = torch.FloatTensor(
            np.array(xs)).permute(0, 3, 1, 2).to(self.device)

        xs = self.net.backbone(xs)
        scores = self.net.head(self.kernel, xs)
        scores = scores.squeeze(1).cpu().numpy()

        # 三次插值upscale，加上缩放scale的惩罚
        scores_up = [cv2.resize(score, (cfg.upscale_size, cfg.upscale_size), 
            interpolation=cv2.INTER_CUBIC) * cfg.scale_penalty for score in scores]
        scores_up[cfg.scale_num // 2] /= cfg.scale_penalty  # 未缩放的没有惩罚

        # 归一化后混合余弦窗口惩罚
        scale_id = np.argmax(np.amax(scores_up, axis=(1, 2)))
        scores_up = scores_up[scale_id]
        scores_up -= scores_up.min()
        scores_up /= scores_up.sum() + 1e-12
        scores_up = (1 - cfg.window_influence) * scores_up + \
            cfg.window_influence * self.cos_window

        # 将最大值对应坐标转换到原图中
        xy = np.unravel_index(np.argmax(scores_up), scores_up.shape)
        xy_in_scoreup = np.array(xy) - (cfg.upscale_size - 1) / 2  # 原点转换到中央
        xy_in_score = xy_in_scoreup / cfg.upscale_size * cfg.score_size
        xy_in_instance = xy_in_score * cfg.stride
        xy_in_img = xy_in_instance / cfg.instance_size * self.x_size * \
            self.scale_factors[scale_id]
        self.center += xy_in_img[::-1]  # 注意,img的shape是(h, w)

        # 更新scale
        scale = (1 - cfg.scale_lr) * 1. + cfg.scale_lr * \
            self.scale_factors[scale_id]
        self.z_size *= scale
        self.x_size *= scale
        self.target_size *= scale

        return np.array([
            self.center[0] - (self.target_size[0] - 1) / 2,
            self.center[1] - (self.target_size[1] - 1) / 2,
            self.target_size[0], self.target_size[1]
        ])
    
    """
    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()
        box0 = box
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = cfg.upscale_size
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = cfg.scale_step ** np.linspace(
            -(cfg.scale_num // 2),
            cfg.scale_num // 2, cfg.scale_num)

        # exemplar and search sizes
        context = cfg.context_amount * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            cfg.instance_size / cfg.exemplar_size
        
        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        # z = crop_and_resize(
        #     img, self.center, self.z_sz,
        #     out_size=cfg.exemplar_size,
        #     border_value=self.avg_color)
        z, _ = crop_resize_box(img, box0, cfg.context_amount, 
            cfg.exemplar_size, cfg.exemplar_size)
        
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        self.kernel = self.net.backbone(z)
        self.kernel = self.kernel.repeat(cfg.scale_num, 1, 1, 1)
    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        # x = [crop_and_resize(
        #     img, self.center, self.x_sz * f,
        #     out_size=cfg.instance_size,
        #     border_value=self.avg_color) for f in self.scale_factors]
        x = [crop_resize_center(
            img, (self.center[1], self.center[0]), 
            (self.x_sz * s, self.x_sz * s),
            (cfg.instance_size, cfg.instance_size))
            for s in self.scale_factors]
        # x = np.stack(x, axis=0)
        # x = torch.from_numpy(x).to(
        #     self.device).permute(0, 3, 1, 2).float()
        x = torch.FloatTensor(
            np.array(x)).permute(0, 3, 1, 2).to(self.device)
        
        # responses
        x = self.net.backbone(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        # responses = np.stack([cv2.resize(
        #     u, (self.upscale_sz, self.upscale_sz),
        #     interpolation=cv2.INTER_CUBIC)
        #     for u in responses])
        # responses[:cfg.scale_num // 2] *= cfg.scale_penalty
        # responses[cfg.scale_num // 2 + 1:] *= cfg.scale_penalty
        
        # 三次插值upscale，加上缩放scale的惩罚
        responses = [cv2.resize(score, (cfg.upscale_size, cfg.upscale_size), 
            interpolation=cv2.INTER_CUBIC) * cfg.scale_penalty for score in responses]
        responses[cfg.scale_num // 2] /= cfg.scale_penalty  # 未缩放的没有惩罚


        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - cfg.window_influence) * response + \
            cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            cfg.stride / 16  #cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / cfg.instance_size
        self.center += disp_in_image

        # update target size
        scale =  (1 - cfg.scale_lr) * 1.0 + \
            cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
"""

    # 重写track函数是因为GOT-10k库中使用PIL图像，不兼容
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            start_time = time.time()
            if f == 0:
                self.init(image, box)
            else:
                boxes[f, :] = self.update(image)
            times[f] = time.time() - start_time

            if visualize:
                # 为了与库中的方法兼容转成PIL的格式再送进去
                show_frame(Image.fromarray(image), boxes[f, :])

        return boxes, times
