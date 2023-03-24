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
import kornia
import kornia.augmentation as K
from einops import rearrange


NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_dataset(config, split='train'):
    dataset_name = config.dataset.name
    query_params = {
        'query_size': config.dataset.query_size,
    }
    clip_params = {
        'fine_size': config.dataset.clip_size_fine,
        'coarse_size': config.dataset.clip_size_coarse,
        'clip_num_frames': config.dataset.clip_num_frames,
        'sampling': config.dataset.clip_sampling,
        'frame_interval': config.dataset.frame_interval,
    }
    if dataset_name == 'ego4d_vq2d':
        dataset = QueryVideoDataset(
            dataset_name=dataset_name,
            query_params=query_params,
            clip_params=clip_params,
            split=split,
            clip_reader=config.dataset.clip_reader
        )
    return dataset


def process_data(config, sample, split='train', device='cuda'):
    '''
    sample: 
        'clip': clip,                           # [B,T,3,H,W]
        'clip_with_bbox': clip_with_bbox,       # [B,T], binary value 0 / 1
        'clip_bbox': clip_bbox,                 # [B,T,4]
        'query': query                          # [B,3,H2,W2]
    '''
    # if split == 'train':
    #     # each loaded item includes a positive and a negative sample concatenated into length T
    #     B, T, _, H, W = sample['clip'].shape
    #     B, _, H2, W2 = sample['query'].shape
    #     sample['clip'] = rearrange(sample['clip'], 'b (n t) c h w -> (b n) t c h w', n=2, t=T//2)
    #     sample['clip_with_bbox'] = rearrange(sample['clip_with_bbox'], 'b (n t) -> (b n) t', n=2, t=T//2)
    #     sample['clip_bbox'] = rearrange(sample['clip_bbox'], 'b (n t) 4 -> (b n) t 4', n=2, t=T//2)
    #     sample['query'] = rearrange(sample['query'].unsqueeze(1).repeat(1,2,1,1,1),
    #                                 'b n c h w -> (b n) c h w')
    
    B, T, _, H, W = sample['clip'].shape
    B, _, H2, W2 = sample['query'].shape

    normalization = kornia.enhance.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    # normalize the input clips
    sample['clip_origin'] = sample['clip'].clone()
    clip = rearrange(sample['clip'], 'b t c h w -> (b t) c h w').to(device)
    clip = normalization(clip).detach().cpu()
    sample['clip'] = rearrange(clip, '(b t) c h w -> b t c h w', b=B, t=T)

    # augment the query
    sample['query_origin'] = sample['query'].clone()
    if split == 'train':
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
        transform_query =  K.AugmentationSequential(
                #K.ColorJiggle(brightness, contrast, saturation, hue=0.0, p=prob_color),
                K.ColorJitter(brightness, contrast, saturation, hue=0.0, p=prob_color),
                K.RandomHorizontalFlip(p=prob_flip),
                K.RandomResizedCrop((query_size, query_size), scale=(crop_sacle, 1.0), ratio=(crop_ratio_min, crop_ratio_max), p=prob_crop),
                K.RandomAffine(affine_degree, [affine_translate, affine_translate], [affine_scale_min, affine_scale_max], 
                                [affine_shear_min, affine_shear_max], p=prob_affine),
                data_keys=["input"],  # Just to define the future input here.
                same_on_batch=False,
                )
        query = transform_query(sample['query'].to(device))
    # normalize input query
    query = normalization(query)
    sample['query'] = query.detach().cpu()
    return sample


def normalize_bbox(bbox, h, w):
    '''
    bbox torch tensor in shape [4] or [N,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1: # [N,4]
        bbox_cp[:, 0] /= h
        bbox_cp[:, 1] /= w
        bbox_cp[:, 2] /= h
        bbox_cp[:, 3] /= w
        return bbox_cp
    else:
        return torch.tensor([bbox_cp[0]/h, bbox_cp[1]/w, bbox_cp[2]/h, bbox_cp[3]/w])

def recover_bbox(bbox, h, w):
    '''
    bbox torch tensor in shape [4] or [N,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1: # [N,4]
        bbox_cp[:, 0] *= h
        bbox_cp[:, 1] *= w
        bbox_cp[:, 2] *= h
        bbox_cp[:, 3] *= w
        return bbox_cp
    else:
        return torch.tensor([bbox_cp[0]*h, bbox_cp[1]*w, bbox_cp[2]*h, bbox_cp[3]*w])

