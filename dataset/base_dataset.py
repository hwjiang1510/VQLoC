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
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from dataset import dataset_utils

split_files = {
            'train': 'vq_train.json',
            'val': 'vq_val.json',            # there is no test
            'test': 'vq_test_unannotated.json'
        }

NORMALIZE_MEAN = [int(it*255) for it in [0.485, 0.456, 0.406]]
NORMALIZE_STD = [int(it*255) for it in [0.229, 0.224, 0.225]]


class QueryVideoDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 query_params,
                 clip_params,
                 data_dir='/vision/hwjiang/episodic-memory/VQ2D/data',
                 clip_dir='/vision/vision_data/Ego4D/v1/clips_320p',
                 meta_dir='/vision/hwjiang/episodic-memory/VQ2D/data',
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
        anno_processed_path = os.path.join('./data', '{}_anno_new.json'.format(self.split))
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


    def _get_video_path(self, sample):
        video_name = sample['metadata']['video_uid']
        video_path = os.path.join(self.video_dir, video_name + '.mp4')
        return video_path
    
    
    def _get_clip_path(self, sample):
        clip_name = sample['clip_uid']
        clip_path = os.path.join(self.clip_dir, clip_name + '.mp4')
        return clip_path
    

    def _get_query_path(self, sample):
        clip_name = sample['clip_uid'] + '-' + sample['query_set']
        image_name = int(sample["visual_crop"]["frame_number"])# "{}/frame_{:07d}.jpg"
        # image_path = os.path.join(self.data_dir, 'images_hw', "{}/frame_{:07d}.jpg".format(clip_name, image_name + 1))
        image_path = os.path.join(self.data_dir, 'images', "{}/frame_{:07d}.png".format(clip_name, image_name + 1))
        return image_path


    def _get_video_lens(self):
        video_len_list = {}
        for idx, cur_anno in enumerate(self.annotations):
            video_path = self._get_video_path(cur_anno)
            video_len_list[video_path] = get_video_len(video_path)
            #video_len_list.append(get_video_len(video_path))
        return video_len_list
    

    def _get_clip_bbox(self, sample, clip_idxs):
        clip_with_bbox, clip_bbox = [], []
        response_track = sample['response_track']
        clip_bbox_all = {}
        for it in response_track:
            clip_bbox_all[int(it['frame_number'])] = [it['y'], it['x'], it['y'] + it['height'], it['x'] + it['width']] # in torch
            origin_hw = [int(it['original_height']), int(it['original_width'])]
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
    

    def _get_query(self, sample, query_path):
        query = Image.open(query_path)
        width, height = query.size
        # validate image size
        anno_width = sample["visual_crop"]["original_width"]
        anno_height = sample["visual_crop"]["original_height"]
        if (anno_height, anno_width) != (height, width):
            query = query.resize((anno_width, anno_height))
            width, height = anno_width, anno_height
        # load bounding box, to get VQ crop
        bbox = get_bbox_from_data(sample["visual_crop"])     # BoxMode.XYXY_ABS, for crop only
        if self.query_params['query_square']:
            bbox = dataset_utils.bbox_cv2Totorch(torch.tensor(bbox))
            bbox = dataset_utils.create_square_bbox(bbox, height, width)
            bbox = dataset_utils.bbox_torchTocv2(bbox).tolist()
        # crop image to get query
        query = query.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        # pad query
        if self.query_params['query_padding']:
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
            query_size = self.query_params['query_size']
            query = F.interpolate(query.unsqueeze(0), size=(query_size, query_size), mode='bilinear').squeeze(0)
        else:
            query_size = self.query_params['query_size']
            query = query.resize((query_size, query_size))
            query = torch.from_numpy(np.asarray(query) / 255.0).permute(2,0,1)  # RGB, [C,H,W]
        return query
    

    def _get_query_frame(self, sample, query_path):
        target_size = self.clip_params['fine_size']
        query = Image.open(query_path)
        width, height = query.size
        # validate image size
        anno_width = sample["visual_crop"]["original_width"]
        anno_height = sample["visual_crop"]["original_height"]
        if (anno_height, anno_width) != (height, width):
            query = query.resize((anno_width, anno_height))
            width, height = anno_width, anno_height
        # load bounding box, to get VQ crop
        bbox = get_bbox_from_data(sample["visual_crop"])     # BoxMode.XYXY_ABS, [4]
        bbox = dataset_utils.bbox_cv2Totorch(torch.tensor(bbox))
        if self.query_params['query_square']:
            bbox = dataset_utils.create_square_bbox(bbox, height, width)
        w, h = query.size
        max_size, min_size = max(h, w), min(h, w)
        pad_height = True if h < w else False
        pad_size = (max_size - min_size) // 2
        if pad_height:
            pad_input = [0, pad_size] * 2                   # for the left, top, right and bottom borders respectively
            bbox[0] += (max_size - min_size) / 2.0   # in padded image size
            bbox[2] += (max_size - min_size) / 2.0
        else:
            pad_input = [pad_size, 0] * 2
            bbox[1] += (max_size - min_size) / 2.0
            bbox[3] += (max_size - min_size) / 2.0  
        transform_pad = transforms.Pad(pad_input, fill=self.padding_value)
        query = transform_pad(query)        # square image
        query = query.resize((target_size, target_size))
        query = torch.from_numpy(np.asarray(query) / 255.0).permute(2,0,1)  # RGB, [C,H,W]
        bbox = bbox / float(max_size)                # in range [0,1]
        return query, bbox
    

    def _get_query_train(self, clip, clip_bbox, clip_with_bbox, query_canonical):
        '''
        clip: [T,3,H,W], value range [0,1]
        clip_bbox: [T,4], in torch axis, value range [0,1]
        clip_with_bbox: [T]
        '''
        h,w = clip.shape[-2:]
        #try:
        fg_idxs = torch.where(clip_with_bbox)[0].numpy().tolist()
        idx = random.choice(fg_idxs)
        
        frame = (clip[idx] * 255).permute(1,2,0).numpy().astype(np.uint8)
        frame = Image.fromarray(frame)
        bbox = dataset_utils.recover_bbox(clip_bbox[idx], h, w)
        bbox = dataset_utils.bbox_torchTocv2(bbox).tolist()
        query = frame.crop((bbox[0], bbox[1], bbox[2], bbox[3]))

        query_size = self.query_params['query_size']
        query = query.resize((query_size, query_size))
        query = torch.from_numpy(np.asarray(query) / 255.0).permute(2,0,1)
        # except:
        #     query = query_canonical
        return query
    
    def _process_bbox(self, clip_bbox, clip_with_bbox, min_size=0.05, max_ratio=2.5):
        '''
        clip_bbox in shape [T,4], value within [0,1], xyxy in torch coordinate
        clip_with_bbox in shape [T], float
        '''
        T = clip_bbox.shape[0]
        min_ratio = 1 / max_ratio

        clip_bbox_h = clip_bbox[:,2] - clip_bbox[:,0]   # [T]
        clip_bbox_w = clip_bbox[:,3] - clip_bbox[:,1]   # [T]

        # clean the annotation by bbox size
        clip_with_bbox *= (clip_bbox_w > min_size).float()
        clip_with_bbox *= (clip_bbox_h > min_size).float()

        # clean the annotation by bbox width
        clip_bbox_ratio = clip_bbox_h / clip_bbox_w
        clip_with_bbox *= (clip_bbox_ratio < max_ratio).float()
        clip_with_bbox *= (clip_bbox_ratio > min_ratio).float()

        return clip_bbox, clip_with_bbox
    

    def _process_clip(self, clip, clip_bbox, clip_with_bbox):
        '''
        clip: in [T,C,H,W]
        bbox: in [T,4] with torch coordinate with value range [0,1] normalized
        clip_with_bbox: in [T]
        '''
        target_size = self.clip_params['fine_size']

        t, _, h, w = clip.shape
        clip_bbox = dataset_utils.recover_bbox(clip_bbox, h, w)

        try:
            fg_idxs = torch.where(clip_with_bbox)[0].numpy().tolist()
            idx = random.choice(fg_idxs)
            frame = (clip[idx] * 255).permute(1,2,0).numpy().astype(np.uint8)
            frame = Image.fromarray(frame)
            bbox = dataset_utils.bbox_torchTocv2(clip_bbox[idx]).tolist()
            query = frame.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            query_size = self.query_params['query_size']
            query = query.resize((query_size, query_size))
            query = torch.from_numpy(np.asarray(query) / 255.0).permute(2,0,1)
        except:
            query = None

        max_size, min_size = max(h, w), min(h, w)
        pad_height = True if h < w else False
        pad_size = (max_size - min_size) // 2
        if pad_height:
            pad_input = [0, pad_size] * 2                   # for the left, top, right and bottom borders respectively
            clip_bbox[:,0] += (max_size - min_size) / 2.0   # in padded image size
            clip_bbox[:,2] += (max_size - min_size) / 2.0
        else:
            pad_input = [pad_size, 0] * 2
            clip_bbox[:,1] += (max_size - min_size) / 2.0
            clip_bbox[:,3] += (max_size - min_size) / 2.0
        
        transform_pad = transforms.Pad(pad_input, fill=self.padding_value)
        clip = transform_pad(clip)        # square image
        h_pad, w_pad = clip.shape[-2:]
        clip = F.interpolate(clip, size=(target_size, target_size), mode='bilinear')#.squeeze(0)
        clip_bbox = clip_bbox / float(h_pad)                # in range [0,1]

        # if self.split == 'train':
        #     clip_bbox, clip_with_bbox = self._process_bbox(clip_bbox, clip_with_bbox)
        return clip, clip_bbox, clip_with_bbox, query, h, w

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
            'before_query': before_query.bool(),            # [T]
            'clip_bbox': clip_bbox.float().clamp(min=0.0, max=1.0),                 # [T,4]
            'query': query.float(),                         # [3,H2,W2]
            'clip_h': torch.tensor(clip_h),
            'clip_w': torch.tensor(clip_w),
            'query_frame': query_frame.float(),             # [3,H,W]
            'query_frame_bbox': query_frame_bbox.float()    # [4]
        }
        return results
    

def sample_frames_balance(num_frames, query_frame, frame_interval, sample, sampling='rand'):
    '''
    sample clips with balanced negative and postive samples
    params:
        num_frames: total number of frames to sample
        query_frame: query time index
        frame_interval: frame interval, where value 1 is for no interval (consecutive frames)
        sample: data annotations
        sampling: only effective for frame_interval larger than 1
    return: 
        frame_idxs: length [num_frames]
    '''
    required_len = (num_frames - 1) * frame_interval + 1
    anno_valid_idx_range = sample["response_track_valid_range"]
    anno_len = anno_valid_idx_range[1] - anno_valid_idx_range[0] + 1
    
    if anno_len <= required_len:
        if anno_len < required_len:
            num_valid = anno_len // frame_interval
        else:
            num_valid = num_frames
        num_invalid = num_frames - num_valid
        if anno_valid_idx_range[1] < required_len:
            idx_start = random.choice(range(anno_valid_idx_range[0])) if anno_valid_idx_range[0] > 0 else 0
            idx_end = idx_start + required_len
        else:
            num_prior = random.choice(range(num_invalid)) if num_invalid != 0 else 0
            num_post = num_invalid - num_prior
            idx_start = anno_valid_idx_range[0] - frame_interval * num_prior
            idx_end = anno_valid_idx_range[1] + frame_interval * num_post + 1
        intervals = np.linspace(start=idx_start, stop=idx_end, num=num_frames+1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1]))
        if sampling == 'rand':
            frame_idxs_pos = [random.choice(range(x[0], x[1])) for x in ranges]
        elif sampling == 'uniform':
            frame_idxs_pos = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        num_addition = anno_len - required_len
        start = random.choice(range(num_addition))
        frame_idxs_pos = [anno_valid_idx_range[0] + start + it for it in range(num_frames)]
    return frame_idxs_pos


decord.bridge.set_bridge("torch")

def read_frames_decord_balance(video_path, num_frames, frame_interval, sample, sampling='rand'):
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    origin_fps = int(video_reader.get_avg_fps())
    gt_fps = int(sample['clip_fps'])
    down_rate = origin_fps // gt_fps
    query_frame = int(sample['query_frame'])
    frame_idxs = sample_frames_balance(num_frames, query_frame, frame_interval, sample, sampling)      # downsampled fps idxs, used to get bbox annotation
    before_query = torch.tensor(frame_idxs) < query_frame
    frame_idxs_origin = [min(it * down_rate, vlen - 1) for it in frame_idxs]        # origin clip fps frame idxs
    #video_reader.skip_frames(1)
    frames = video_reader.get_batch(frame_idxs_origin)
    frames = frames.float() / 255
    frames = frames.permute(0, 3, 1, 2)
    return frames, frame_idxs, before_query

def get_bbox_from_data(data):
    # BoxMode.XYXY_ABS
    return [data["x"], data["y"], data["x"] + data["width"], data["y"] + data["height"]]

def get_video_len(video_path):
    cap = cv2.VideoCapture(video_path)
    if not (cap.isOpened()):
        return False
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return vlen

video_reader_dict = {
    'decord_balance': read_frames_decord_balance,
}