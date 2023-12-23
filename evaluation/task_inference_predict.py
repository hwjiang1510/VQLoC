import time
import os
import torch
import decord
import cv2
from PIL import Image
from dataset import dataset_utils
from evaluation.test_dataloader import load_query, load_clip, process_inputs
from einops import rearrange
from utils import vis_utils
from utils.loss_utils import GiouLoss

NMS_IOU = 0.65

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
        self.clip_dir = config.clip_dir

    def run(self, model, config, device):
        clip_uid = self.annots[0]["clip_uid"]

        if clip_uid is None:
            return 
        clip_path = os.path.join(self.clip_dir, clip_uid  + '.mp4')
        if not os.path.exists(clip_path):
            print(f"Clip {clip_uid} does not exist")
            return 

        for key, annot in zip(self.keys, self.annots):
            annotation_uid = annot["metadata"]["annotation_uid"]
            query_set = annot["metadata"]["query_set"]
            annot_key = f"{annotation_uid}_{query_set}"
            query_frame = annot["query_frame"]
            visual_crop = annot["visual_crop"]
            save_path = os.path.join(self.config.inference_cache_path, f'{annot_key}.pt')
            if os.path.isfile(save_path):
                continue

            ret_bboxes, ret_scores = inference_video(config, 
                                                     model, 
                                                     clip_path, 
                                                     query_frame, 
                                                     visual_crop,
                                                     os.path.join(self.config.inference_cache_path, annot_key),
                                                     device)
            save_dict = {'ret_bboxes': ret_bboxes,
                         'ret_scores': ret_scores}
            torch.save(save_dict, save_path)



def inference_video(config, model, clip_path, query_frame, visual_crop, save_path, device):
    '''
    Perform VQ2D inference:
        1. Load the query crop and clip
        2. Batchify the clips
        3. Do inference
    '''
    clip_reader = decord.VideoReader(clip_path, num_threads=1)
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]
    oshape = (owidth, oheight)

    # get query
    query = load_query(config, clip_reader, visual_crop, clip_path)     # [c,h,w]

    # get clip frames
    clip_reader = decord.VideoReader(clip_path, num_threads=1)
    origin_fps = int(clip_reader.get_avg_fps())
    assert origin_fps == 5
    vlen = len(clip_reader)
    search_window = list(range(0, query_frame))
    if query_frame >= vlen:
        print("=====> WARNING: Going out of range. Clip path: {}, Len: {}, query_frame: {}".format(
                clip_path, len(clip_reader), query_frame))

    clip_num_frames = config.dataset.clip_num_frames    # 30
    batch_size = config.train.batch_size
    batch_num_frames = clip_num_frames * batch_size

    inference_time = (query_frame - 1) // batch_num_frames
    if (query_frame - 1) % batch_num_frames != 0:
        inference_time += 1
    
    ret_bboxes, ret_scores = [], []
    for i in range(inference_time):
        # get the batch size for the current inference
        idx_start = min(i * batch_num_frames, query_frame-1)
        idx_end = min((i+1) * batch_num_frames, query_frame-1)
        num_frames = idx_end - idx_start
        if num_frames < batch_num_frames:
            num_frames += 1
        batch_size_inference = num_frames // clip_num_frames
        if num_frames % clip_num_frames != 0:
            batch_size_inference += 1
        assert batch_size_inference <= batch_size

        # index padding
        inference_num_frames = batch_size_inference * clip_num_frames
        frame_idx = list(range(idx_start, idx_end))
        if len(frame_idx) < inference_num_frames:
            num_pad = inference_num_frames - len(frame_idx)
            frame_idx.extend([idx_end] * num_pad)   # [N=B*T]
        #print(query_frame, idx_start, idx_end, num_frames, min(frame_idx), max(frame_idx))
        
        # get current clips
        clips_origin, clips = load_clip(config, clip_reader, frame_idx, clip_path)    # [N,3,H,W]
        clips_origin = clips_origin[:num_frames]
        clips = rearrange(clips, '(b t) c h w -> b t c h w', b=batch_size_inference, t=clip_num_frames)

        # process inputs
        clips = clips.to(device).float()
        query = query.to(device).float()
        clips_raw = clips.clone()
        query_raw = query.clone()
        try:
            clips, queries = process_inputs(clips, query)
        except:
            print(clips.shape, idx_start, idx_end, batch_size_inference, inference_num_frames, query_frame, inference_time)
        clips = clips.to(device)
        queries = queries.to(device)

        # inference
        with torch.no_grad():
            preds = model(clips, queries, fix_backbone=config.model.fix_backbone)
        preds_top = get_top_predictions(config, preds, num_frames, oshape)
        ret_bboxes.append(preds_top['bbox'])
        ret_scores.append(preds_top['prob'])

        if config.debug: #and device == torch.device("cuda:0"):
            vis_utils.vis_pred_clip_inference(clips=clips_origin, 
                                    queries=query_raw,
                                    pred=preds_top,
                                    save_path=save_path,
                                    iter_num=i)

    ret_bboxes = torch.cat(ret_bboxes, dim=0)
    ret_scores = torch.cat(ret_scores, dim=0)
    return ret_bboxes, ret_scores


def get_top_predictions(config, preds, num_frames, oshape):
    '''
    preds with shape [b,t,N,...] or [b,t,...], N is number of anchor box
    '''
    owidth, oheight = oshape    # origin resolution of clips
    resize_res = config.dataset.clip_size_coarse

    pred_center = preds['center']   # [b,t,N,2] or [b,t,2]
    pred_hw = preds['hw']           # [b,t,N,2], actually half of hw
    pred_bbox = preds['bbox']       # [b,t,N,4]
    pred_prob = preds['prob']       # [b,t,N]

    if len(pred_prob.shape) == 3:
        # with anchor
        b,t,N = pred_prob.shape
        pred_prob = rearrange(pred_prob, 'b t N -> (b t) N')
        pred_hw = rearrange(pred_hw, 'b t N c -> (b t) N c')
        pred_center = rearrange(pred_center, 'b t N c -> (b t) N c')
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t) N c')
        pred_prob_all = pred_prob.clone()

        pred_prob, top_idx = torch.max(pred_prob, dim=-1)  # [b*t], [b*t]
        pred_bbox = torch.gather(pred_bbox, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,4)).squeeze()       # [b*t,4]
        pred_hw = torch.gather(pred_hw, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)).squeeze()           # [b*t,2]
        pred_center = torch.gather(pred_center, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)).squeeze()   # [b*t,2]

    else:
        b,t = pred_prob.shape
        pred_prob = rearrange(pred_prob, 'b t -> (b t)')
        pred_hw = rearrange(pred_hw, 'b t c -> (b t) c')
        pred_center = rearrange(pred_center, 'b t c -> (b t) c')
        pred_bbox = rearrange(pred_bbox, 'b t c -> (b t) c')

        #iou = process_prob(top_idx, pred_prob, preds)    # [b*t,N]
    pred_prob_raw = pred_prob.clone().detach().cpu()
    pred_bbox_raw = pred_bbox.clone().detach().cpu()
    pred_prob = pred_prob[:num_frames]
    pred_bbox = pred_bbox[:num_frames]
    pred_bbox_processed = process_bbox_prediction(pred_bbox, owidth, oheight, resize_res)
    
    preds = {
        'bbox_raw': pred_bbox_raw,
        'prob_raw': pred_prob_raw,
        'bbox': pred_bbox_processed.detach().cpu(),
        'prob': pred_prob.detach().cpu(),
        # 'anchor_iou_top': iou[:num_frames].detach().cpu(),
        # 'prob_all': pred_prob_all[:num_frames].detach().cpu()
    }
    return preds


def process_prob(top_idx, top_prob, preds):
    '''
    top_idx in shape [b*t]
    top_prob in shape [b*t]
    '''
    pred_bbox = preds['bbox']       # [b,t,N,4]
    pred_prob = preds['prob']       # [b,t,N]
    b,t,N = pred_prob.shape

    pred_prob = rearrange(pred_prob, 'b t N -> (b t) N')        # [b*t,N]
    pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t) N c')    # [b*t,N,4]

    top_bbox = torch.gather(pred_bbox, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,4)).squeeze()  # [b*t,4]
    top_bbox = top_bbox.reshape(-1,1,4)     # [b*t,1,4]

    iou = get_iou(top_bbox, pred_bbox)  # [b*t,N]
    return iou

    # mean_prob = []
    # for i in range(b*t):
    #     cur_iou = iou[i]    # [N]
    #     cur_prob = pred_prob[i]
    #     cur_iou_mask = cur_iou > NMS_IOU
    #     if cur_iou_mask.sum() > 0:
    #         NMS_prob = cur_prob[cur_iou_mask].mean()
    #     else:
    #         NMS_prob = top_prob[i]
    #     #print(torch.sigmoid(top_prob[i]).item(), cur_iou.mean().item(), cur_iou_mask.sum().item(), torch.sigmoid(NMS_prob).item())
    #     mean_prob.append(NMS_prob)
    # mean_prob = torch.tensor(mean_prob)     # [b*t]
    # return mean_prob


def process_bbox_prediction(pred_bbox, owidth, oheight, resize_res):
    '''
    pred_bbox, in shape [N,4], value in [0,1], corresponding to resize_res resolution, x1y1x2y2 in torch axis
    process the bounding box by:
        1. clamp the value of padded region
        2. return to original resolution
        3. turn to BoxMode.XYXY_ABS (cv2 axis)
    '''
    max_size, min_size = max(owidth, oheight), min(owidth, oheight)
    diff_size = max_size - min_size
    diff_size_ratio = diff_size / max_size
    diff_size_ratio_half = diff_size_ratio / 2.0

    if owidth >= oheight:
        width_min, width_max = 0.0, 1.0
        height_min, height_max = diff_size_ratio_half, 1.0 - diff_size_ratio_half
    else:
        width_min, width_max = diff_size_ratio_half, 1.0 - diff_size_ratio_half
        height_min, height_max = 0.0, 1.0

    x1, y1, x2, y2 = pred_bbox.split([1,1,1,1], dim=-1)     # [N,1]
    x1 = (x1 - height_min) / (height_max - height_min)
    y1 = (y1 - width_min) / (width_max - width_min)
    x2 = (x2 - height_min) / (height_max - height_min)
    y2 = (y2 - width_min) / (width_max - width_min)
    pred_bbox = torch.cat([x1, y1, x2, y2], dim=-1)     # [N,4], in range [0,1]
    pred_bbox = pred_bbox.clamp(min=0.0, max=1.0)

    pred_bbox = dataset_utils.recover_bbox(pred_bbox, oheight, owidth)
    pred_bbox = dataset_utils.bbox_torchTocv2(pred_bbox)

    return pred_bbox


def get_iou(top_bbox, pred_bbox):
    '''
    top_bbox in shape [b*t,1,4]
    pred_bbox in shape [b*t,N,4]
    '''
    B,N,_ = pred_bbox.shape

    top_bbox_replicate = top_bbox.repeat(1,N,1)     # [b*t,N,4]
    
    pred_bbox = pred_bbox.reshape(-1,4)
    top_bbox_replicate = top_bbox_replicate.reshape(-1,4)

    iou, giou, loss_iou = GiouLoss(pred_bbox, top_bbox_replicate)
    iou = iou.reshape(B, N)
    return iou


