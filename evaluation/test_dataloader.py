import time
import os
import torch
import decord
import cv2
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import kornia
from dataset.dataset_utils import NORMALIZE_MEAN, NORMALIZE_STD
from einops import rearrange

from dataset.base_dataset import get_bbox_from_data
from dataset import dataset_utils


def load_query(config, clip_reader, visual_crop, clip_path):
    '''
    return: in shape [c,h,w] RGB
    '''
    # load query frame
    vc_fno = visual_crop["frame_number"]
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]
    if vc_fno >= len(clip_reader):
        print(
            "=====> WARNING: Going out of range. Clip path: {}, Len: {}, j: {}".format(
                clip_path, len(clip_reader), vc_fno
            )
        )
    query = clip_reader.get_batch([vc_fno])[0].numpy()  # RGB format, [h,w,3]
    if (query.shape[0] != oheight) or (query.shape[1] != owidth):
        query = cv2.resize(query, (owidth, oheight))
    query = Image.fromarray(query)

    # load bounding box annotation and process
    bbox = get_bbox_from_data(visual_crop)     # BoxMode.XYXY_ABS, for crop only
    if config.dataset.query_square:
        bbox = dataset_utils.bbox_cv2Totorch(torch.tensor(bbox))
        bbox = dataset_utils.create_square_bbox(bbox, oheight, owidth)
        bbox = dataset_utils.bbox_torchTocv2(bbox).tolist()

    # crop image to get query
    query = query.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

    # pad query
    if config.dataset.query_padding:
        transform = transforms.Compose([transforms.ToTensor()])
        query = transform(query)    # [c,h,w]
        _, h, w = query.shape
        max_size, min_size = max(h, w), min(h, w)
        pad_height = True if h < w else False
        pad_size = (max_size - min_size) // 2
        if pad_height:
            pad_input = [0, pad_size] * 2                   # for the left, top, right and bottom borders respectively
        else:
            pad_input = [pad_size, 0] * 2
        transform_pad = transforms.Pad(pad_input)
        query = transform_pad(query)        # square image
        # resize query
        query_size = config.dataset.query_size
        query = F.interpolate(query.unsqueeze(0), size=(query_size, query_size), mode='bilinear').squeeze(0)
    else:
        query_size = config.dataset.query_size
        query = query.resize((query_size, query_size))
        query = torch.from_numpy(np.asarray(query) / 255.0).permute(2,0,1)  # RGB, [c,h,w]
    return query


def load_clip(config, clip_reader, frame_idx, clip_path):
    '''
    frame_idx: list of N elements of index
    return: loaded video frames in shape [N,3,h,w]
    '''
    if config.dataset.padding_value == 'zero':
        pad_value = 0
    elif config.dataset.padding_value == 'mean':
        pad_value = 0.5

    clips = clip_reader.get_batch(frame_idx)
    clips = clips.float() / 255
    clips = clips.permute(0, 3, 1, 2)     # [N,3,h,w]
    clips_origin = clips.clone()

    # resize and pad
    target_size = config.dataset.clip_size_coarse
    N, _, h, w = clips.shape
    max_size, min_size = max(h, w), min(h, w)
    pad_height = True if h < w else False
    pad_size = (max_size - min_size) // 2
    if pad_height:
        pad_input = [0, pad_size] * 2                   # for the left, top, right and bottom borders respectively
    else:
        pad_input = [pad_size, 0] * 2
    transform_pad = transforms.Pad(pad_input, pad_value)
    clips = transform_pad(clips)        # square image
    h_pad, w_pad = clips.shape[-2:]
    clips = F.interpolate(clips, size=(target_size, target_size), mode='bilinear')
    return clips_origin, clips


def process_inputs(clips, query):
    '''
    clips: in shape [b,t,c,h,w]
    query: in shape [c,h,w]
    '''
    b, t, _, h, w = clips.shape
    normalization = kornia.enhance.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    clips = rearrange(clips, 'b t c h w -> (b t) c h w')
    clips = normalization(clips)
    clips = rearrange(clips, '(b t) c h w -> b t c h w', b=b, t=t)

    queries = normalization(query.unsqueeze(0).repeat(b,1,1,1)) # [b,c,h,w]
    return clips, queries
