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
from dataset.base_dataset import QueryVideoDataset, video_reader_dict

split_files = {
            'train': 'vq_train.json',
            'val': 'vq_val.json',            # there is no test
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
                 ):
        self.dataset_name = dataset_name
        self.query_params = query_params
        self.clip_params = clip_params
        
        self.data_dir = data_dir
        self.clip_dir = clip_dir
        self.meta_dir = meta_dir
        self.video_dir = '/vision/vision_data/Ego4D/v1/full_scale'

        self.split = split

        self.clip_reader = video_reader_dict[clip_reader]
        self._load_metadata()


    def _load_metadata(self):
        anno_processed_path = os.path.join('./data', '{}_egotracks_anno.json'.format(self.split))
        if os.path.isfile(anno_processed_path):
            with open(anno_processed_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            os.makedirs('./data', exist_ok=True)
            target_split_fp = split_files[self.split]
            ann_file = os.path.join(self.meta_dir, target_split_fp)
            with open(ann_file) as f:
                anno_json = json.load(f)

            self.annotations, n_samples, n_samples_valid = [], 0, 0
            for video_data in anno_json['videos']:
                for clip_data in video_data['clips']:
                    for clip_anno in clip_data['annotations']:
                        for qset_id, qset in clip_anno['query_sets'].items():
                            if not qset['is_valid']:
                                continue
                            response_track_frame_ids = []
                            for frame_it in qset['response_track']:
                                response_track_frame_ids.append(int(frame_it['frame_number']))
                            frame_id_min, frame_id_max = min(response_track_frame_ids), max(response_track_frame_ids)
                            curr_anno = {
                                "metadata": {
                                    "video_uid": video_data["video_uid"],
                                    "video_start_sec": clip_data["video_start_sec"],
                                    "video_end_sec": clip_data["video_end_sec"],
                                    "clip_fps": clip_data["clip_fps"],
                                },
                                "clip_uid": clip_data["clip_uid"],
                                "clip_fps": clip_data["clip_fps"],
                                "query_set": qset_id,
                                "query_frame": qset["query_frame"],
                                "response_track": sorted(qset["response_track"], key=lambda x: x['frame_number']),
                                "response_track_valid_range": [frame_id_min, frame_id_max], 
                                "visual_crop": qset["visual_crop"],
                                "object_title": qset["object_title"],
                                # Assign a unique ID to this annotation for the dataset
                                "dataset_uid": f"{self.split}_{n_samples_valid:010d}"
                            }
                            query_path = self._get_query_path(curr_anno)
                            if clip_data["clip_uid"] == '859ed253-d752-4f1b-adc3-c76599117d6e':
                                print(query_path)
                            if os.path.isfile(query_path):
                                self.annotations.append(curr_anno)
                                n_samples_valid += 1
                            elif self.split == 'train' and clip_data["clip_uid"] == '859ed253-d752-4f1b-adc3-c76599117d6e':
                                print(query_path, curr_anno['clip_uid'], curr_anno['visual_crop']['frame_number'], curr_anno['visual_crop'])
                            n_samples += 1
            print('Find {} data samples, {} valid (query path exist)'.format(n_samples, n_samples_valid))
            with open(anno_processed_path, 'w') as ff:
                json.dump(self.annotations, ff)
                        
        print('Data split {}, with {} samples'.format(self.split, len(self.annotations)))
    