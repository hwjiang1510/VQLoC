import os
import pdb

import tqdm
import random
import json

import cv2
import decord
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from dataset import dataset_utils
from dataset.base_dataset import QueryVideoDataset, read_frames_decord_balance, get_bbox_from_data, get_video_len
from dataset.base_dataset import NORMALIZE_MEAN, NORMALIZE_STD

split_files = {
            # 'train': 'egotracks_train.json',
            # 'val': 'egotracks_val.json',            # there is no test
            # 'test': 'egotracks_challenge_test_unannotated.json'
            'train': 'vq_train.json',
            'val': 'vq_val.json',
            'test': 'vq_test_unannotated.json'
        }


class EgoTracksDataset(QueryVideoDataset):
    def __init__(self,
                 dataset_name,
                 query_params,
                 clip_params,
                 data_dir='/vision/hwjiang/episodic-memory/VQ2D/data',
                 clip_dir='/vision/vision_data/Ego4D/v1/clips_320p',
                 meta_dir='/vision/hwjiang/episodic-memory/egotrack/data',
                 split='train',
                 clip_reader='decord_balance',
                 eval_vis_freq=50,
                 ):
        self.dataset_name = dataset_name
        self.query_params = query_params
        self.clip_params = clip_params

        if self.clip_params['padding_value'] == 'zero':
            self.padding_value = 0
        elif self.clip_params['padding_value'] == 'mean':
            self.padding_value = 0.5 #tuple(NORMALIZE_MEAN)
        
        self.data_dir = data_dir
        self.clip_dir = clip_dir
        self.meta_dir = meta_dir
        self.video_dir = '/vision/vision_data/Ego4D/v1/full_scale'

        self.split = split

        self.clip_reader = video_reader_dict[clip_reader]
        self._load_metadata()
        if self.split != 'train':
            self.annotations = self.annotations[::eval_vis_freq]


    def _load_metadata(self):
        # anno_processed_path = os.path.join('./data', '{}_egotracks_anno_1.json'.format(self.split))
        # if os.path.isfile(anno_processed_path):
        #     # print("!!!!!!!!!!!!!!!!!!!!!")
        #     with open(anno_processed_path, 'r') as f:
        #         self.annotations = json.load(f)
        # else:
        os.makedirs('./data', exist_ok=True)
        target_split_fp = split_files[self.split]
        ann_file = os.path.join(self.meta_dir, target_split_fp)
        # print("target split file: ", target_split_fp)
        # print("meta_dir", self.meta_dir)
        # print("ann_file!!!!!!!!!", ann_file)
        # exit()
        assert os.path.isfile(ann_file)
        with open(ann_file) as f:
            anno_json = json.load(f)

        self.annotations, n_samples, n_samples_valid = [], 0, 0
        # for video_data in anno_json['videos']:
        #     for clip_data in video_data['clips']:
        # print(len(anno_json))
        # print(anno_json[0])
        for clip_uid in anno_json.keys():
            for queries in anno_json[clip_uid]['annotations']:
                # for clip_anno in query_set['annotations']:
                for qset_id, qset in queries['query_sets'].items():
                    if not qset['is_valid']:
                        continue
                    response_track_frame_ids = []
                    for frame_it in qset['response_track']:
                        response_track_frame_ids.append(int(frame_it['frame_number']))
                    frame_id_min, frame_id_max = min(response_track_frame_ids), max(response_track_frame_ids)
                    if 'lt_track' in qset.keys():
                        lt_track = qset['lt_track']
                    else:
                        lt_track = qset['response_track']
                        # n_no_lt_track += 1
                    lt_track_frame_ids = []
                    for frame_it in lt_track:
                        lt_track_frame_ids.append(int(frame_it['frame_number']))
                    curr_anno = {
                        "metadata": {
                            "video_uid": clip_uid,
                            # "video_start_sec": clip_data["video_start_sec"],
                            # "video_end_sec": clip_data["video_end_sec"],
                            "clip_fps": 5,
                        },
                        "clip_uid": clip_uid,
                        "clip_fps": 5,
                        "query_set": qset_id,
                        "query_frame": qset["query_frame"],
                        "response_track": sorted(qset["response_track"], key=lambda x: x['frame_number']),
                        "response_track_valid_range": [frame_id_min, frame_id_max], 
                        "lt_track": sorted(lt_track, key=lambda x: x['frame_number']),
                        "lt_track_frame_ids": sorted(lt_track_frame_ids),
                        "visual_crop": qset["visual_crop"],
                        "object_title": qset["object_title"],
                        # Assign a unique ID to this annotation for the dataset
                        "dataset_uid": f"{self.split}_{n_samples_valid:010d}"
                    }
                    query_path = self._get_query_path(curr_anno)
                    # if clip_uid == '859ed253-d752-4f1b-adc3-c76599117d6e':
                        # print(query_path)
                    if os.path.isfile(query_path):
                        self.annotations.append(curr_anno)
                        n_samples_valid += 1
                    elif self.split == 'train':
                        print(query_path, curr_anno['clip_uid'], curr_anno['visual_crop']['frame_number'], curr_anno['visual_crop'])
                    n_samples += 1
        print('Find {} data samples, {} valid (query path exist)'.format(n_samples, n_samples_valid))
        # with open(anno_processed_path, 'w') as ff:
        #     json.dump(self.annotations, ff)
        print('Data split {}, with {} samples'.format(self.split, len(self.annotations)))

    def _get_origin_hw_clip(self, response_track):
        for it in response_track:
            origin_hw = [int(it['original_height']), int(it['original_width'])]
            return origin_hw
    
    def _get_clip_bbox(self, sample, clip_idxs):
        clip_with_bbox, clip_bbox = [], []
        origin_hw = self._get_origin_hw_clip(sample['response_track'])
        response_track = sample['lt_track']
        clip_bbox_all = {}
        for it in response_track:
            clip_bbox_all[int(it['frame_number'])] = [it['y'], it['x'], it['y'] + it['height'], it['x'] + it['width']] # in torch
            #origin_hw = [int(it['original_height']), int(it['original_width'])]
        for id in clip_idxs:
            if int(id) in clip_bbox_all.keys():
                clip_with_bbox.append(True)
                cur_bbox = torch.tensor(clip_bbox_all[int(id)])
                cur_bbox_normalize = dataset_utils.normalize_bbox(cur_bbox, origin_hw[0], origin_hw[1])
                clip_bbox.append(cur_bbox_normalize)
            else:
                clip_with_bbox.append(False)
                clip_bbox.append(torch.tensor([0.0, 0.0, 0.00001, 0.00001]))
        clip_with_bbox = torch.tensor(clip_with_bbox).float()    # [T]
        clip_bbox = torch.stack(clip_bbox)                      # [T, 4]
        return clip_with_bbox, clip_bbox
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sample = self.annotations[idx]
        video_path = self._get_video_path(sample)
        query_path = self._get_query_path(sample)
        clip_path = self._get_clip_path(sample)

        sample_method = self.clip_params['sampling']
        if self.clip_reader == 'decord_balance':
            assert sample_method == 'rand'
        if self.split == 'test':
            sample_method = 'uniform'

        # load clip, in shape [T,C,H,W] within value range [0,1]
        try:
            if os.path.isfile(clip_path):
                clip, clip_idxs, before_query = self.clip_reader(clip_path, 
                                                   self.clip_params['clip_num_frames'],
                                                   self.clip_params['frame_interval'],
                                                   sample,
                                                   sampling=sample_method)
                                                #    sample_method, 
                                                #    fix_start=fix_start)
            else:
                print(f"Warning: missing video file {clip_path}.")
                assert False
        except Exception as e:
                raise ValueError(
                    f'Clip loading failed for {clip_path}, clip loading for this dataset is strict.') from e
        
        # load clip bounding box
        clip_with_bbox, clip_bbox = self._get_clip_bbox(sample, clip_idxs)

        # clip with square shape, bbox processed accordingly
        clip, clip_bbox, clip_with_bbox, query, clip_h, clip_w = self._process_clip(clip, clip_bbox, clip_with_bbox)

        # load query image
        query_canonical = self._get_query(sample, query_path)
        #if self.split != 'train' or (not torch.is_tensor(query)):
        query = query_canonical.clone()

        # load original query frame and the bbox
        query_frame, query_frame_bbox = self._get_query_frame(sample, query_path)

        results = {
            'clip': clip.float(),                           # [T,3,H,W]
            'clip_with_bbox': clip_with_bbox.float(),       # [T]
            'before_query': torch.ones_like(clip_with_bbox).bool(), #before_query.bool(),            # [T]
            'clip_bbox': clip_bbox.float().clamp(min=0.0, max=1.0),                 # [T,4]
            'query': query.float(),                         # [3,H2,W2]
            'clip_h': torch.tensor(clip_h),
            'clip_w': torch.tensor(clip_w),
            'query_frame': query_frame.float(),             # [3,H,W]
            'query_frame_bbox': query_frame_bbox.float()    # [4]
        }
        return results


decord.bridge.set_bridge("torch")

def sample_frames_random(num_frames, query_frame, frame_interval, sample, sampling):
    '''
    sample clips with balanced negative and postive samples
    params:
        num_frames: total number of frames to sample
        query_frame: query time index
        frame_interval: frame interval, where value 1 is for no interval (consecutive frames)
        sample: data annotations
        sampling: only effective for frame_interval larger than 1
    return: 
        frame_idxs: length [2*num_frames]
    '''
    assert frame_interval == 1

    lt_tracks = sample['lt_track']      # include multiple tracks for egotrack
    lt_track_frame_ids = sample['lt_track_frame_ids']
    lt_track_frame_ids_largest = max(lt_track_frame_ids)

    rt_track_frame_idx_range = sample["response_track_valid_range"]
    rt_track_frame_idx_max = rt_track_frame_idx_range[1]

    idx = random.choice(lt_track_frame_ids)
    num_frames_left = random.choice(list(range(num_frames)))
    idx_left = max(0, idx - num_frames_left + 1)
    idx_right = idx_left + num_frames

    selected_idx = list(range(idx_left, idx_right))
    cur_num_frames = idx_right - idx_left
    if cur_num_frames < num_frames:
        selected_idx += [selected_idx[-1]] * (num_frames - len(selected_idx))

    # if (query_frame - rt_track_frame_idx_max) > num_frames:
    #     num = torch.rand(1).item()
    #     if num > 0.3:
    #         num_frames_left = query_frame - rt_track_frame_idx_max - num_frames
    #         idx_start = random.choice(list(range(num_frames_left)))
    #         idx_start += rt_track_frame_idx_max
    #         idx_end = idx_start + num_frames
    #         selected_idx = list(range(idx_start, idx_end))

    return selected_idx


def read_frames_decord_random(video_path, num_frames, frame_interval, sample, sampling='rand'):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    origin_fps = int(video_reader.get_avg_fps())
    gt_fps = int(sample['clip_fps'])
    down_rate = origin_fps // gt_fps
    query_frame = int(sample['query_frame'])
    frame_idxs = sample_frames_random(num_frames, query_frame, frame_interval, sample, sampling)      # downsampled fps idxs, used to get bbox annotation
    before_query = torch.tensor(frame_idxs) < query_frame
    frame_idxs_origin = [min(it * down_rate, vlen - 1) for it in frame_idxs]        # origin clip fps frame idxs
    #video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs_origin)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, before_query
    

video_reader_dict = {
    'decord_balance': read_frames_decord_balance,
    'decord_random': read_frames_decord_random,
}
