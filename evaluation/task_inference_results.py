import time
import os
import torch
import decord
import cv2
import numpy as np
from PIL import Image
from dataset import dataset_utils
from evaluation.test_dataloader import load_query, load_clip, process_inputs
from einops import rearrange
from utils import vis_utils
import scipy
from scipy.signal import find_peaks, medfilt
from evaluation.structures import BBox, ResponseTrack
import random


SMOOTHING_SIGMA = 5
DISTANCE = 25
WIDTH = 3
PROMINENCE = 0.2
PEAK_SCORE_THRESHILD = 0.5  
PEAK_WINDWOW_RATIO = 0.5

PEAK_SCORE_THRESHOLD = 0.8
PEAK_WINDOW_THRESHOLD = 0.7


class Task:
    def __init__(self, config, annots):
        super().__init__()
        self.config = config
        self.annots = annots
        # Ensure that all annotations belong to the same clip
        clip_uid = annots[0]["clip_uid"]
        for annot in self.annots:
            assert annot["clip_uid"] == clip_uid
        self.keys = [
            (annot["metadata"]["annotation_uid"], annot["metadata"]["query_set"])
            for annot in self.annots
        ]
        # self.clip_dir = '/vision/srama/Research/Ego4D/episodic-memory/VQ2D/data/clips_fullres'
        # self.clip_dir = '../dlcv/DLCV_vq2d_data/clips'
        self.clip_dir = config.clip_dir

    def run(self, config, device):
        clip_uid = self.annots[0]["clip_uid"]
        if clip_uid is None:
            print(self.annots[0]["metadata"]["annotation_uid"])
            latest_bbox_format = [BBox(0, 0.0, 0.0, 0.0, 0.0)]
            all_pred_rts = {}
            for key, annot in zip(self.keys, self.annots):
                pred_rts = [ResponseTrack(latest_bbox_format, score=1.0)]
                all_pred_rts[key] = pred_rts
            return all_pred_rts

        clip_path = os.path.join(self.clip_dir, clip_uid  + '.mp4')
        if not os.path.exists(clip_path):
            print(f"Clip {clip_uid} does not exist")
            return {}

        all_pred_rts = {}
        for key, annot in zip(self.keys, self.annots):
            annotation_uid = annot["metadata"]["annotation_uid"]
            query_set = annot["metadata"]["query_set"]
            annot_key = f"{annotation_uid}_{query_set}"
            query_frame = annot["query_frame"]
            visual_crop = annot["visual_crop"]
            save_path = os.path.join(self.config.inference_cache_path, f'{annot_key}.pt')
            assert os.path.isfile(save_path)
            cache = torch.load(save_path)
            ret_bboxes, ret_scores = cache['ret_bboxes'], torch.sigmoid(cache['ret_scores'])
            ret_bboxes = ret_bboxes.numpy()     # bbox in [N,4], original resolution, cv2 axis
            ret_scores = ret_scores.numpy()     # scores in [N]

            ret_scores_sm = ret_scores.copy()
            for i in range(1):
                ret_scores_sm = medfilt(ret_scores_sm, kernel_size=SMOOTHING_SIGMA)

            # only used for testing stAP with gt window 
            # gt_scores = np.zeros_like(ret_scores_sm)
            # len_clip = gt_scores.shape[0]
            # gt_rt_idx = [int(frame_it['frame_number']) for frame_it in annot['response_track']]
            # for frame_it in gt_rt_idx:
            #     gt_scores[min(frame_it, len_clip-1)] = random.uniform(0.6,1)
            # ret_scores_sm = gt_scores.copy()

            peaks, _ = find_peaks(ret_scores_sm)
            if len(peaks) == 0:
                print(ret_scores_sm)
            peaks = process_peaks(peaks, ret_scores_sm)

            recent_peak = None
            for peak in peaks[::-1]:
                recent_peak = peak
                break
            # print(ret_scores_sm[recent_peak])

            if recent_peak is not None:
                threshold = ret_scores_sm[recent_peak] * PEAK_WINDOW_THRESHOLD
                latest_idx = [recent_peak]
                for idx in range(recent_peak, 0, -1):
                    if ret_scores_sm[idx] >= threshold:
                        latest_idx.append(idx)
                    else:
                        break
                for idx in range(recent_peak, query_frame-1):
                    if ret_scores_sm[idx] >= threshold:
                        latest_idx.append(idx)
                    else:
                        break
            else:
                latest_idx = [query_frame-2]
            
            latest_idx = sorted(list(set(latest_idx)))
            latest_bbox = ret_bboxes[latest_idx]    # [t,4]
            score = ret_scores_sm[recent_peak]
            
            latest_bbox_format = []
            for (frame_bbox, fram_idx) in zip(latest_bbox, latest_idx):
                x1, y1, x2, y2 = frame_bbox
                bbox_format = BBox(fram_idx, x1, y1, x2, y2)
                latest_bbox_format.append(bbox_format)
            
            pred_rts = [ResponseTrack(latest_bbox_format, score=score)]
            all_pred_rts[key] = pred_rts
        
        return all_pred_rts


def process_peaks(peaks_idx, ret_scores_sm):
    '''process the peaks based on their scores'''
    num_frames = ret_scores_sm.shape[0]
    if len(peaks_idx) == 0:
        start_score, end_score = ret_scores_sm[0], ret_scores_sm[-1]
        if start_score > end_score:
            valid_peaks_idx = [0]
        else:
            valid_peaks_idx = [num_frames-1]
    else:
        peaks_score = ret_scores_sm[peaks_idx]
        largest_score = np.max(peaks_score)

        threshold = largest_score * PEAK_SCORE_THRESHOLD

        valid_peaks_idx_idx = np.where(peaks_score > threshold)[0]
        valid_peaks_idx = peaks_idx[valid_peaks_idx_idx]
    return valid_peaks_idx

