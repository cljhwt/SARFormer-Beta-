import random

import cv2
import numpy as np
import torch
import copy
from torch import nn, Tensor
import os

import math
import torch.nn.functional as F
from torch import nn
from ...visualization import SegLocalVisualizer
save_dir = './'
seg_local_visualizer = SegLocalVisualizer(
            vis_backends=[dict(type='LocalVisBackend')],
            save_dir=save_dir)
seg_local_visualizer.dataset_meta = dict(
    classes=(
        'wall', 'building', 'sky', 'road', 'sidewalk', 'tree', 'grass', 'water',
        'window', 'door', 'signboard', 'fence', 'car', 'bus', 'motorbike', 'bicycle',
        'person', 'dog', 'cat', 'chair', 'table', 'potted plant', 'TV', 'laptop', 'cell phone',
        'refrigerator', 'microwave', 'oven', 'sink', 'bathtub', 'bed', 'toilet', 'curtain',
        'carpet', 'mirror', 'book', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        'banner', 'food', 'kitchen', 'sky-other', 'ceiling', 'cupboard', 'paper', 'blanket', 'pillow',
        'screen', 'board', 'card', 'whiteboard', 'camera', 'bookshelf', 'clock', 'calendar',
        'refrigerator-other', 'lamp', 'umbrella', 'storage box', 'stereo', 'tableware', 'skateboard',
        'climbing equipment', 'door handle', 'stair', 'countertop', 'step', 'awning', 'blind',
        'stereo set', 'computer', 'utensil', 'truck'
    ),

    palette=[
        [128, 64, 128], [70, 70, 70], [70, 130, 180], [128, 64, 128], [244, 35, 232],
        [0, 102, 0], [107, 142, 35], [0, 0, 70], [102, 102, 156], [190, 153, 153],
        [220, 220, 0], [190, 250, 190], [70, 130, 180], [0, 0, 142], [0, 0, 70],
        [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 32],
        [255, 204, 54], [0, 153, 204], [85, 85, 85], [102, 102, 156], [153, 153, 153],
        [220, 220, 0], [244, 5, 242], [220, 147, 0], [0, 70, 100], [102, 202, 60],
        [204, 153, 204], [0, 20, 230], [153, 0, 153], [255, 165, 0], [153, 153, 153],
        [153, 153, 153], [156, 102, 102], [153, 153, 153], [0, 80, 100], [0, 0, 142],
        [0, 0, 70], [0, 60, 100], [0, 0, 230], [204, 250, 250], [255, 0, 16],
        [64, 0, 128], [102, 0, 204], [190, 153, 153], [0, 204, 255], [255, 51, 51],
        [250, 230, 190], [220, 248, 255], [50, 153, 204], [200, 130, 0], [204, 51, 51],
        [102, 250, 200], [255, 102, 0], [153, 51, 102], [255, 204, 51], [204, 250, 128],
        [0, 102, 0], [0, 204, 51], [204, 204, 255], [250, 204, 50], [255, 153, 204],
        [51, 250, 250], [200, 200, 0], [250, 200, 0], [250, 200, 200], [0, 0, 0],
        [204, 50, 153], [0, 250, 50], [70, 70, 70],
        # 继续添加颜色，直至总数达到 92 个颜色
    ])

def gen_encoder_output_proposals(memory: Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def float_to_mask(x, t):
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    return torch.where(x > t, one, zero)

def ddim_sample(self, backbone_feats, images, images_whwh,clip_denoised=True, do_postprocess=True):
    batch = images_whwh.shape[0]
    h,w=images.image_sizes[0]
    shape = (batch, self.num_queries, h,w)
    total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
    #import pdb;pdb.set_trace()
    # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
    #tensor([ -1., 999.])
    times = list(reversed(times.int().tolist()))

    time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    img = torch.randn(shape, device=self.device)
    x_start = None
    output=[]
    for time, time_next in time_pairs:
        time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
        self_cond = x_start if self.self_condition else None
        #import pdb;pdb.set_trace()
        pred_noise, x_start,outputs= self.model_predictions(backbone_feats, images_whwh, img, time_cond, self_cond, clip_x_start=clip_denoised)
        output.append(outputs)
        if time_next < 0:
            img = x_start
            continue
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]
        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        noise = torch.randn_like(x_start,dtype=torch.float32)

        img = x_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise

def mmseg_to_detectron_gt( batch_data_samples):
    targets = []
    for sample in batch_data_samples:
        gt_sem_seg = sample.gt_sem_seg.data

        # 获取gt_sem_seg中出现的类别
        unique_classes = np.unique(gt_sem_seg.cpu())
        unique_classes = unique_classes[unique_classes != 255]

        labels = torch.tensor(unique_classes)

        masks = []
        for class_idx in unique_classes:
            # 创建与gt_sem_seg相同形状的掩码，值为True或False
            mask = (gt_sem_seg == class_idx)
            masks.append(mask)
        if not masks:  # 如果masks为空，添加一个空张量
            masks.append(torch.empty(0, dtype=torch.bool))
        masks = torch.cat(masks, dim=0)
        targets.append({"labels": labels, "masks": masks})

    return targets

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


# 转换为二进制mask
def float_to_mask(x, t):
    one = torch.ones_like(x)
    zero = torch.zeros_like(x)
    return torch.where(x > t, one, zero)


def show_anns(anns, image, output_path='./vis_data/output_image.png'):
    result_image = image.copy()
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    for ann in sorted_anns:
        m = ann['segmentation']
        # 生成随机颜色
        color_mask = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # 将True部分的mask绘制在结果图像上
        result_image[m] = color_mask
    # 保存结果图像
    cv2.imwrite(output_path, result_image)

def xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]

def is_mask_fully_covered(mask_bbox, box, tolerance=5):
    # 将xywh格式的mask_bbox转换为[x1, y1, x2, y2]格式
    mask_xyxy = xywh_to_xyxy(mask_bbox)
    # 考虑宽容度，判断mask是否被bbox包含
    if (mask_xyxy[0] >= box[0] - tolerance) and (mask_xyxy[1] >= box[1] - tolerance) and \
       (mask_xyxy[2] <= box[2] + tolerance) and (mask_xyxy[3] <= box[3] + tolerance):
        return True
    return False

def mask_filt(shape,boxes,masks_generated):
    if len(boxes)==0:
        return masks_generated
    filtered_masks = []
    for mask in masks_generated:
        # if mask['area']<shape[0]*shape[1]/200:
        #     continue
        cover=0
        for box in boxes:
            if is_mask_fully_covered(mask['bbox'], box):
                cover=1
                break
        if cover==0:
            filtered_masks.append(mask)

    return filtered_masks