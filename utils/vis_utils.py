import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import imageio
import os
from dataset import dataset_utils
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np


def vis_pred_clip(sample, pred, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)

    clip = sample['clip_origin'].detach().cpu()        # [B,T,3,H,W]
    query = sample['query_origin'].detach().cpu()      # [B,3,H2,W2]
    query_aug = sample['query'].detach().cpu()         # [B,3,H2,W2]
    bbox = sample['clip_bbox'].detach().cpu()          # [B,T,4]
    bbox_pred = pred['bbox'].detach().cpu()            # [B,T,4]
    prob = sample['clip_with_bbox'].detach().cpu()     # [B,T]
    prob_pred = pred['prob'].detach().cpu()            # [B,T]

    B, T, _, H, W = clip.shape
    _, _, H2, W2 = query_aug.shape

    for i in range(B):
        frames = []
        cur_clip, cur_query = clip[i], query[i]                                     # [T,3,H,W], [3,H2,W2]
        cur_bbox, cur_bbox_pred = bbox[i], bbox_pred[i].clamp(min=0.0, max=1.0)     # [T,4]
        cur_prob, cur_prob_pred = prob[i], prob_pred[i]                             # [T]

        cur_query = cur_query.clamp(min=0.0, max=1.0).permute(1,2,0).numpy()        # [H2,W2,3]
        for j in range(T):
            # draw clips with bbox
            img = cur_clip[j].clamp(min=0.0, max=1.0)                               
            img = img.permute(1,2,0).numpy()                # [H,W,3]
            fig, ax = plt.subplots(1,2, dpi=100)
            fig.suptitle('Prob: gt {:.3f}, pred {:.3f}'.format(cur_prob[j].item(), torch.sigmoid(cur_prob_pred[j]).item()), fontsize=20)
            ax[0].imshow(img)
            ax[1].imshow(cur_query)
            if cur_prob[j].item() > 0.5:
                draw_bbox_gt = dataset_utils.recover_bbox(cur_bbox[j], H, W)  # [4]
                rect = patches.Rectangle((draw_bbox_gt[1], draw_bbox_gt[0]), 
                                         draw_bbox_gt[3]-draw_bbox_gt[1], draw_bbox_gt[2]-draw_bbox_gt[0], 
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax[0].add_patch(rect)
            if cur_prob[j].item() > 0.5:
                draw_bbox_pred = dataset_utils.recover_bbox(cur_bbox_pred[j], H, W)  # [4]
                rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                                         draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                                         linewidth=1, edgecolor='g', facecolor='none')
                ax[0].add_patch(rect)
            if torch.sigmoid(cur_prob_pred[j]).item() > 0.5:
                draw_bbox_pred = dataset_utils.recover_bbox(cur_bbox_pred[j], H, W)  # [4]
                rect = patches.Rectangle((draw_bbox_pred[1], draw_bbox_pred[0]), 
                                         draw_bbox_pred[3]-draw_bbox_pred[1], draw_bbox_pred[2]-draw_bbox_pred[0], 
                                         linewidth=1, edgecolor='b', facecolor='none')
                ax[0].add_patch(rect)
            plt.savefig(os.path.join(output_dir, 'tmp.png'))
            plt.close()
            frames.append(cv2.imread(os.path.join(output_dir, 'tmp.png'))[...,::-1])
        save_name = os.path.join(output_dir, '{}_{}.gif'.format(iter_num, i))
        imageio.mimsave(save_name, frames, 'GIF', duration=0.2)


def vis_pred_scores(sample, pred, iter_num, output_dir, subfolder='train'):
    output_dir = os.path.join(output_dir, 'visualization', subfolder)
    os.makedirs(output_dir, exist_ok=True)

    prob = sample['clip_with_bbox'].detach().cpu()     # [B,T]
    prob_pred = pred['prob'].detach().cpu()            # [B,T]
    if 'gt_iou' in pred.keys():
        prob_iou = pred['gt_iou'].detach().cpu()            # [B,T]
    if 'prob_refine' in pred.keys():
        prob_refine = pred['prob_refine'].detach().cpu()            # [B,T]
    B, T = prob.shape

    for i in range(B):
        cur_prob, cur_prob_pred = prob[i].numpy(), torch.sigmoid(prob_pred[i]).numpy()     # [T]
        x = np.arange(T)
        plt.plot(x, cur_prob_pred, marker=None, color='b', label='pred')
        plt.plot(x, cur_prob, marker=None, color='r', label='gt')
        if 'prob_refine' in pred.keys():
            cur_prob_refine = torch.sigmoid(prob_refine[i]).numpy()
            plt.plot(x, cur_prob_refine, marker=None, color='g', label='pred')
        if 'gt_iou' in pred.keys():
            cur_prob_iou = prob_iou[i].numpy() * 0.9
            plt.plot(x, cur_prob_iou, marker=None, color='c', label='pred')
        plt.xlabel('number of frames')
        plt.ylabel('occurance score')
        plt.ylim((0.0, 1.05))
        plt.legend(loc='best')
        save_name = os.path.join(output_dir, '{}_{}.jpg'.format(iter_num, i))
        plt.savefig(save_name)
        plt.close()


def vis_pred_clip_inference(clips, queries, pred, save_path, iter_num):
    #clips = clips.detach().cpu()            # [b,t,c,h,w]
    queries = queries.detach().cpu()        # [c,h,w]
    # bbox = pred['bbox_raw']                 # [b*t,4]
    # prob = torch.sigmoid(pred['prob_raw'])  # [b*t]
    bbox = pred['bbox']                 # [b*t,4]
    prob = torch.sigmoid(pred['prob'])  # [b*t]
    save_name = save_path + f'_{iter_num}.mp4'
    writer = imageio.get_writer(save_name, fps=5)

    #clips = rearrange(clips, 'b t c h w -> (b t) c h w')

    T, _, H, W = clips.shape
    _, H2, W2 = queries.shape

    frames = []
    for i in range(T):
        cur_clip = clips[i].clamp(min=0.0, max=1.0).permute(1,2,0).numpy()
        cur_query = queries.clamp(min=0.0, max=1.0).permute(1,2,0).numpy()
        cur_bbox = bbox[i]#.clamp(min=0.0, max=1.0)
        cur_prob = prob[i]

        fig, ax = plt.subplots(1,2)
        fig.suptitle('Prob {:.3f}'.format(cur_prob.item()), fontsize=20)
        ax[0].imshow(cur_clip)
        ax[1].imshow(cur_query)
        if cur_prob.item() > 0.5:
            draw_bbox_pred = cur_bbox #dataset_utils.recover_bbox(cur_bbox, H, W)  # [4]
            rect = patches.Rectangle((draw_bbox_pred[0], draw_bbox_pred[1]), 
                                      draw_bbox_pred[2]-draw_bbox_pred[0], draw_bbox_pred[3]-draw_bbox_pred[1], 
                                      linewidth=1, edgecolor='b', facecolor='none')
            ax[0].add_patch(rect)
        plt.savefig(save_path + '_tmp.jpg')
        plt.close()
        writer.append_data(cv2.imread(save_path + '_tmp.jpg')[...,::-1])
    writer.close()



