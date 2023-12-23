import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import argparse
from dataset.base_dataset import QueryVideoDataset
from dataset.egotracks_dataset import EgoTracksDataset
import kornia
import kornia.augmentation as K
from kornia.constants import DataKey
from einops import rearrange


NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_dataset(config, split='train'):
    if split == 'train':
        clip_num_frames = config.dataset.clip_num_frames
        clip_reader = config.dataset.clip_reader
        dataset_name = config.dataset.name
    else:
        clip_num_frames = config.dataset.clip_num_frames_val
        clip_reader = config.dataset.clip_reader_val
        dataset_name = config.dataset.name_val

    query_params = {
        'query_size': config.dataset.query_size,
        'query_padding': config.dataset.query_padding,
        'query_square': config.dataset.query_square,
    }
    clip_params = {
        'fine_size': config.dataset.clip_size_fine,
        'coarse_size': config.dataset.clip_size_coarse,
        'clip_num_frames': clip_num_frames,
        'sampling': config.dataset.clip_sampling,
        'frame_interval': config.dataset.frame_interval,
        'padding_value': config.dataset.padding_value
    }

    if dataset_name == 'ego4d_vq2d':
        dataset = QueryVideoDataset(
            dataset_name=dataset_name,
            query_params=query_params,
            clip_params=clip_params,
            split=split,
            clip_reader=clip_reader
        )
    elif dataset_name == 'ego4d_egotracks':
        dataset = EgoTracksDataset(
            dataset_name=dataset_name,
            query_params=query_params,
            clip_params=clip_params,
            # data_dir='../Meta/data',
            # clip_dir='../Meta/DLCV_vq2d_data/clips',
            # meta_dir='../Meta/DLCV_vq2d_data',
            data_dir=config.data_dir,
            clip_dir=config.clip_dir,
            meta_dir=config.meta_dir,
            split=split,
            clip_reader=clip_reader
        )
    return dataset


def process_data(config, sample, iter=0, split='train', device='cuda'):
    '''
    sample: 
        'clip': clip,                           # [B,T,3,H,W]
        'clip_with_bbox': clip_with_bbox,       # [B,T], binary value 0 / 1
        'clip_bbox': clip_bbox,                 # [B,T,4]
        'query': query                          # [B,3,H2,W2]
    '''    
    B, T, _, H, W = sample['clip'].shape
    B, _, H2, W2 = sample['query'].shape
    normalization = kornia.enhance.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    brightness = config.train.aug_brightness
    contrast = config.train.aug_contrast
    saturation = config.train.aug_saturation
    query_size = config.dataset.query_size
    crop_sacle = config.train.aug_crop_scale
    crop_ratio_min = config.train.aug_crop_ratio_min
    crop_ratio_max = config.train.aug_crop_ratio_max
    affine_degree = config.train.aug_affine_degree
    affine_translate = config.train.aug_affine_translate
    affine_scale_min = config.train.aug_affine_scale_min
    affine_scale_max = config.train.aug_affine_scale_max
    affine_shear_min = config.train.aug_affine_shear_min
    affine_shear_max = config.train.aug_affine_shear_max
    prob_color = config.train.aug_prob_color
    prob_flip = config.train.aug_prob_flip
    prob_crop = config.train.aug_prob_crop
    prob_affine = config.train.aug_prob_affine

    transform_clip = K.AugmentationSequential(
                K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0, p=1.0),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomResizedCrop((H, W), scale=(0.66, 1.0), ratio=(crop_ratio_min, crop_ratio_max), p=1.0),
                # K.RandomAffine(affine_degree, [affine_translate, affine_translate], [affine_scale_min, affine_scale_max], 
                #                 [affine_shear_min, affine_shear_max], p=prob_affine),
                data_keys=[DataKey.INPUT, DataKey.BBOX_XYXY],  # Just to define the future input here.
                same_on_batch=True,
                )
    transform_query = K.AugmentationSequential(
                K.ColorJitter(brightness, contrast, saturation, hue=0, p=prob_color),
                K.RandomHorizontalFlip(p=prob_flip),
                K.RandomResizedCrop((query_size, query_size), scale=(crop_sacle, 1.0), ratio=(crop_ratio_min, crop_ratio_max), p=prob_crop),
                # K.RandomAffine(affine_degree, [affine_translate, affine_translate], [affine_scale_min, affine_scale_max], 
                #                 [affine_shear_min, affine_shear_max], p=prob_affine
                # K.RandomAffine(affine_degree, [0, 0], [1.0, 1.0], 
                #                 [1.0, 1.0], p=prob_affine),
                data_keys=["input"],  # Just to define the future input here.
                same_on_batch=False,
                )
    
    transform_query_frame = K.AugmentationSequential(
                K.ColorJitter(brightness, contrast, saturation, hue=0, p=prob_color),
                K.RandomHorizontalFlip(p=prob_flip),
                # K.RandomAffine(affine_degree, [affine_translate, affine_translate], [affine_scale_min, affine_scale_max], 
                #                 [affine_shear_min, affine_shear_max], p=prob_affine
                # K.RandomAffine(affine_degree, [0, 0], [1.0, 1.0], 
                #                 [1.0, 1.0], p=prob_affine),
                data_keys=[DataKey.INPUT, DataKey.BBOX_XYXY],  # Just to define the future input here.
                same_on_batch=False,
                )
    
    clip = sample['clip']                           # [B,T,C,H,W]
    query = sample['query']                         # [B,C,H',W']
    clip_with_bbox = sample['clip_with_bbox']       # [B,T]
    clip_bbox = sample['clip_bbox']                 # [B,T,4], with value range [0,1], torch axis
    clip_bbox = recover_bbox(clip_bbox, H, W)       # [B,T,4], with range in image pixels, torch axis
    clip_bbox = bbox_torchTocv2(clip_bbox)          # [B,T,4], with range in image pixels, cv2 axis
    if config.train.use_query_roi and 'query_frame' in sample.keys():
        query_frame = sample['query_frame']                         # [B,C,H,W]
        query_frame_bbox = sample['query_frame_bbox']   
        query_frame_bbox = recover_bbox(query_frame_bbox, H, W)
        query_frame_bbox = bbox_torchTocv2(query_frame_bbox)        # [B,4]

    # augment clips
    if split == 'train' and config.train.aug_clip and (iter > config.train.aug_clip_iter):        
        clip_aug, clip_bbox_aug  = [], []
        for clip_cur, clip_bbox_cur in zip(clip, clip_bbox):    # [T,C,H,W], [T,4,2]
            clip_cur_aug, clip_bbox_cur_aug = transform_clip(clip_cur.to(device), clip_bbox_cur.to(device).unsqueeze(1))
            clip_aug.append(clip_cur_aug)
            clip_bbox_aug.append(clip_bbox_cur_aug.squeeze())

        clip_aug = torch.stack(clip_aug)                     # [B,T,C,H,W]
        clip_bbox_aug = torch.stack(clip_bbox_aug)           # [B,T,4]
        clip_bbox_aug = bbox_cv2Totorch(clip_bbox_aug)
        clip_bbox_aug, with_bbox_update = check_bbox(clip_bbox_aug, H, W)
        clip_bbox_aug = normalize_bbox(clip_bbox_aug, H, W)                 # back in range [0,1]
        clip_with_bbox_aug = torch.logical_and(with_bbox_update.to(clip_with_bbox.device), clip_with_bbox)
        sample['clip'] = clip_aug.to(device)
        sample['clip_with_bbox'] = clip_with_bbox_aug.to(device).float()
        sample['clip_bbox'] = clip_bbox_aug.to(device)
    
    # augment the query
    if split == 'train' and config.train.aug_query:
        query = transform_query(query)
        sample['query'] = query.to(device)
    
    # augment the query frame
    if split == 'train' and config.train.aug_query and 'query_frame' in sample.keys():
        query_frame, query_frame_bbox = transform_query_frame(query)
        sample['query_frame'] = query_frame.to(device)
        query_frame_bbox = bbox_cv2Totorch(query_frame_bbox)
        query_frame_bbox = normalize_bbox(query_frame_bbox, H, W).clamp(min=0.0, max=1.0)
        sample['query_frame_bbox'] = query_frame_bbox.to(device).float()

    # normalize the input clips
    sample['clip_origin'] = sample['clip'].clone()
    clip = rearrange(sample['clip'], 'b t c h w -> (b t) c h w').to(device)
    clip = normalization(clip)
    sample['clip'] = rearrange(clip, '(b t) c h w -> b t c h w', b=B, t=T)

    # normalize input query
    sample['query_origin'] = sample['query'].clone()
    sample['query'] = normalization(sample['query'])

    # normalize input query frame
    if 'query_frame' in sample.keys():
        sample['query_frame_origin'] = sample['query_frame'].clone()
        sample['query_frame'] = normalization(sample['query_frame'])

    return sample


def replicate_sample_for_hnm(gts):
    '''
        gts = {
            'clip':                 in [b,t,c,h,w]
            'clip_rigin':           in [b,t,c,h,w]
            'clip_with_bbox':       in [b,t]
            'before_query':         in [b,t]
            'clip_bbox':            in [b,t,4]
            'query':                in [b,c,h,w]
            'query_origin':         in [b,c,h,w]
            'clip_h':               in [b]
            'clip_w':               in [b]
        }
    '''
    clip = gts['clip']
    clip_origin = gts['clip_origin']
    clip_with_bbox = gts['clip_with_bbox']
    before_query = gts['before_query']
    clip_bbox = gts['clip_bbox']
    query = gts['query']
    query_origin = gts['query_origin']
    clip_h, clip_w = gts['clip_h'], gts['clip_w']

    b, t = clip.shape[:2]
    device = clip.device

    new_clip = []
    new_clip_origin = []
    new_clip_with_bbox = []
    new_before_query = []
    new_clip_bbox = []
    new_query = []
    new_query_origin = []
    new_clip_h, new_clip_w = [], []

    for i in range(b):
        for j in range(b):
            new_clip.append(clip[i])
            new_clip_origin.append(clip_origin[i])
            new_query.append(query[j])
            new_query_origin.append(query_origin[j])
            if i == j:
                new_clip_with_bbox.append(clip_with_bbox[i])
                new_before_query.append(before_query[i])
                new_clip_bbox.append(clip_bbox[i])
            else:
                new_clip_with_bbox.append(torch.zeros(t).float().to(device))
                new_before_query.append(torch.ones(t).bool().to(device))
                new_clip_bbox.append(torch.tensor([[0.0, 0.0, 0.0001, 0.0001]]).repeat(t,1).float().to(device))
            new_clip_h.append(clip_h[i])
            new_clip_w.append(clip_w[i])
    
    new_clip = torch.stack(new_clip)
    new_clip_origin = torch.stack(new_clip_origin)
    new_clip_with_bbox = torch.stack(new_clip_with_bbox)
    new_before_query = torch.stack(new_before_query)
    new_clip_bbox = torch.stack(new_clip_bbox)
    new_clip_h = torch.stack(new_clip_h)
    new_clip_w = torch.stack(new_clip_w)
    new_query = torch.stack(new_query)
    new_query_origin = torch.stack(new_query_origin)

    new_gts = {
            'clip': new_clip,                       # in [b^2,t,c,h,w]
            'clip_origin': new_clip_origin,         # in [b^2,t,c,h,w]
            'clip_with_bbox': new_clip_with_bbox,   # in [b^2,t]
            'before_query': new_before_query,       # in [b^2,t]
            'clip_bbox': new_clip_bbox,             # in [b^2,t,4]
            'query': new_query,                     # in [b^2,c,h,w]
            'query_origin': new_query_origin,       # in [b^2,c,h,w]
            'clip_h': new_clip_h,                   # in [b^2]
            'clip_w': new_clip_w,                   # in [b^2]
        }
    return new_gts


def normalize_bbox(bbox, h, w):
    '''
    bbox torch tensor in shape [4] or [...,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1: # [N,4]
        bbox_cp[..., 0] /= h
        bbox_cp[..., 1] /= w
        bbox_cp[..., 2] /= h
        bbox_cp[..., 3] /= w
        return bbox_cp
    else:
        return torch.tensor([bbox_cp[0]/h, bbox_cp[1]/w, bbox_cp[2]/h, bbox_cp[3]/w])


def recover_bbox(bbox, h, w):
    '''
    bbox torch tensor in shape [4] or [...,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1: # [N,4]
        bbox_cp[..., 0] *= h
        bbox_cp[..., 1] *= w
        bbox_cp[..., 2] *= h
        bbox_cp[..., 3] *= w
        return bbox_cp
    else:
        return torch.tensor([bbox_cp[0]*h, bbox_cp[1]*w, bbox_cp[2]*h, bbox_cp[3]*w])
    

def bbox_torchTocv2(bbox):
    '''
    torch, idx 0/2 for height, 1/3 for width (x,y,x,y)
    cv2: idx 0/2 for width, 1/3 for height (y,x,y,x)
    bbox torch tensor in shape [4] or [...,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1:
        bbox_x1 = bbox_cp[...,0].unsqueeze(-1)
        bbox_y1 = bbox_cp[...,1].unsqueeze(-1)
        bbox_x2 = bbox_cp[...,2].unsqueeze(-1)
        bbox_y2 = bbox_cp[...,3].unsqueeze(-1)
        return torch.cat([bbox_y1, bbox_x1, bbox_y2, bbox_x2], dim=-1)
    else:
        return torch.tensor([bbox_cp[1], bbox_cp[0], bbox_cp[3], bbox_cp[2]])
    

def bbox_cv2Totorch(bbox):
    '''
    torch, idx 0/2 for height, 1/3 for width (x,y,x,y)
    cv2: idx 0/2 for width, 1/3 for height (y,x,y,x)
    bbox torch tensor in shape [4] or [...,4], under cv2 axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1:
        bbox_x1 = bbox_cp[...,1].unsqueeze(-1)
        bbox_y1 = bbox_cp[...,0].unsqueeze(-1)
        bbox_x2 = bbox_cp[...,3].unsqueeze(-1)
        bbox_y2 = bbox_cp[...,2].unsqueeze(-1)
        return torch.cat([bbox_x1, bbox_y1, bbox_x2, bbox_y2], dim=-1)
    else:
        return torch.tensor([bbox_cp[1], bbox_cp[0], bbox_cp[3], bbox_cp[2]])


def check_bbox(bbox, h, w):
    B, T, _ = bbox.shape
    bbox = bbox.reshape(-1,4)

    x1, y1, x2, y2 = bbox[...,0], bbox[...,1], bbox[...,2], bbox[...,3]
    left_invalid = y2 <= 0.0
    right_invalid = y1 >= w - 1
    top_invalid = x2 <= 0.0
    bottom_invalid = x1 >= h - 1

    x_invalid = torch.logical_or(top_invalid, bottom_invalid)
    y_invalid = torch.logical_or(left_invalid, right_invalid)
    invalid = torch.logical_or(x_invalid, y_invalid)
    valid = ~invalid

    x1_clip = x1.clip(min=0.0, max=h).unsqueeze(-1)
    x2_clip = x2.clip(min=0.0, max=h).unsqueeze(-1)
    y1_clip = y1.clip(min=0.0, max=w).unsqueeze(-1)
    y2_clip = y2.clip(min=0.0, max=w).unsqueeze(-1)
    bbox_clip = torch.cat([x1_clip, y1_clip, x2_clip, y2_clip], dim=-1)

    return bbox_clip.reshape(B,T,4), valid.reshape(B,T)


def check_bbox_permute(bbox_p):
    '''
    bbox_p: [N,4], (x1,y1,x2,y2)
    '''
    x1p = torch.minimum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    x2p = torch.maximum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    y1p = torch.minimum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
    y2p = torch.maximum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
    bbox_p = torch.cat([x1p, y1p, x2p, y2p], axis=1)
    return bbox_p


def bbox_xyxyTopoints(bbox):
    '''
    bbox: torch.Tensor, in shape [..., 4]
    return: bbox in shape [...,4,2] with 4 points location
    p1---p2
    |     |
    p4---p3
    '''
    bbox_x1 = bbox[...,0].unsqueeze(-1)     # [...,1]
    bbox_y1 = bbox[...,1].unsqueeze(-1)
    bbox_x2 = bbox[...,2].unsqueeze(-1)
    bbox_y2 = bbox[...,3].unsqueeze(-1)

    pt1 = torch.cat([bbox_x1, bbox_y1], dim=-1).unsqueeze(-2)     # [...,1,2]
    pt2 = torch.cat([bbox_x2, bbox_y1], dim=-1).unsqueeze(-2)     # [...,1,2]
    pt3 = torch.cat([bbox_x2, bbox_y2], dim=-1).unsqueeze(-2)     # [...,1,2]
    pt4 = torch.cat([bbox_x1, bbox_y2], dim=-1).unsqueeze(-2)     # [...,1,2]

    pts = torch.cat([pt1, pt2, pt3, pt4], dim=-2)                 # [...,4,2]
    return pts


def bbox_pointsToxyxy(pts):
    '''
    pts: torch.Tensor, in shape [...,4,2]
    return: bbox in shape [...,4] for x1y1x2y2
    '''
    shape_in = list(pts.shape[:-2])
    pts = pts.reshape(-1,4,2)

    pt1 = pts[:,0,:]           # [N,2]
    pt3 = pts[:,3,:]

    x1 = pt1[:, 0].unsqueeze(-1)  # [N,1]
    y1 = pt1[:, 1].unsqueeze(-1)  
    x2 = pt3[:, 0].unsqueeze(-1)  
    y2 = pt3[:, 1].unsqueeze(-1)

    bbox = torch.cat([x1,y1,x2,y2], dim=-1)     # [N,4]
    bbox = bbox.reshape(shape_in + [4])
    return bbox


def create_square_bbox(bbox, img_h, img_w):
    '''
    bbox in [4], in torch coordinate
    '''
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    h = center_x - x1
    w = center_y - y1
    r = max(h, w)

    new_x1 = max(center_x - r, 0)
    new_x2 = min(center_x + r, img_h-1)
    new_y1 = max(center_y - r, 0)
    new_y2 = min(center_y + r, img_w-1)

    new_bbox = torch.tensor([new_x1, new_y1, new_x2, new_y2])
    return new_bbox


def bbox_xyhwToxyxy(bbox_xyhw):
    '''
    bbox_xyhw in shape [..., 4]
    height and width of bbox is the full height and width
    '''
    bbox_center = bbox_xyhw[..., :2]
    bbox_hw = bbox_xyhw[..., 2:]
    bbox_hw_half = 0.5 * bbox_hw

    bbox_xyxy = torch.cat([bbox_center - bbox_hw_half, bbox_center + bbox_hw_half], dim=-1)
    return bbox_xyxy

