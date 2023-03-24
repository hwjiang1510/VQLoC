import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import imageio
import os
from dataset import dataset_utils
import torch.nn.functional as F


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
            fig, ax = plt.subplots(1,2, dpi=200)
            fig.suptitle('Prob: gt {:.3f}, pred {:.3f}'.format(cur_prob[j].item(), F.sigmoid(cur_prob_pred[j]).item()), fontsize=20)
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
            if F.sigmoid(cur_prob_pred[j]).item() > 0.5:
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





