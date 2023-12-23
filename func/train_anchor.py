import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
import dataclasses
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast
import itertools
from utils import exp_utils, train_utils, loss_utils, vis_utils
from dataset import dataset_utils
import wandb
from einops import rearrange
from utils.loss_utils import GiouLoss

logger = logging.getLogger(__name__)


def train_epoch(config, loader, model, optimizer, schedular, scaler, epoch, output_dir, device, rank, wandb_run=None, ddp=True):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()
    
    train_utils.set_model_train(config, model, ddp)

    batch_end = time.time()
    for batch_idx, sample in enumerate(loader):
        iter_num = batch_idx + len(loader) * epoch

        sample = exp_utils.dict_to_cuda(sample)
        sample = dataset_utils.process_data(config, sample, iter=iter_num, split='train', device=device)     # normalize and data augmentations on GPU
        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        # reconstruction loss
        clips, queries = sample['clip'], sample['query']
        #with autocast():
        if config.train.use_query_roi and 'query_frame' in sample.keys():
            preds = model(clips, 
                          sample['query_frame'], 
                          query_frame_bbox=sample['query_frame_bbox'], 
                          training=True, fix_backbone=config.model.fix_backbone)
        else:
            preds = model(clips, queries, training=True, fix_backbone=config.model.fix_backbone)
        time_meters.add_loss_value('Prediction time', time.time() - end)
        end = time.time()

        losses, preds_top, sample = loss_utils.get_losses_with_anchor(config, preds, sample)
        total_loss = 0.0
        for k, v in losses.items():
            if 'loss' in k:
                total_loss += losses[k.replace('loss_', 'weight_')] * v
                loss_meters.add_loss_value(k, v.detach().item())
        total_loss = total_loss / config.train.accumulation_step
        
        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        total_loss.backward()
        if (batch_idx+1) % config.train.accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()
            schedular.step()

        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, all {batch_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].val,
                recon_time=time_meters.average_meters['Prediction time'].val,
                batch_time=time_meters.average_meters['Batch time'].val,
                batch_time_avg=time_meters.average_meters['Batch time'].avg
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)
        
        if iter_num % config.vis_freq == 0 and rank == 0:
            vis_utils.vis_pred_clip(sample=sample,
                                    pred=preds_top,
                                    iter_num=iter_num,
                                    output_dir=output_dir,
                                    subfolder='train')
            vis_utils.vis_pred_scores(sample=sample,
                                    pred=preds_top,
                                    iter_num=iter_num,
                                    output_dir=output_dir,
                                    subfolder='train')

        batch_end = time.time()

        # if rank == 0:
        #     wandb_log = {'Train/loss': total_loss.item(),
        #                 'Train/lr': optimizer.param_groups[0]['lr']}
        #     for k, v in losses.items():
        #         if 'loss' in k:
        #             wandb_log['Train/{}'.format(k)] = v.item()
        #     wandb_run.log(wandb_log)
        
        dist.barrier()
        if batch_idx < 3:
            torch.cuda.empty_cache()



def validate(config, loader, model, epoch, output_dir, device, rank, wandb_run=None, ddp=True):
    model.eval()
    metrics = {}

    with torch.no_grad():
        for batch_idx, sample in enumerate(loader):
            # if batch_idx % config.eval_vis_freq != 0:
            #     continue
            sample = exp_utils.dict_to_cuda(sample)
            sample = dataset_utils.process_data(config, sample, split='val', device=device)     # normalize and data augmentations on GPU

            clips, queries = sample['clip'], sample['query']
            if config.train.use_query_roi and 'query_frame' in sample.keys():
                preds = model(clips, 
                            sample['query_frame'], 
                            query_frame_bbox=sample['query_frame_bbox'], 
                            training=False, fix_backbone=config.model.fix_backbone)
            else:
                preds = model(clips, queries, training=False, fix_backbone=config.model.fix_backbone)
            results, preds_top = val_performance(config, preds, sample)
            try:
                for k, v in results.items():
                    if k in metrics.keys():
                        try:
                            metrics[k].append(v)
                        except:
                            print('1', k, v, metrics[k], batch_idx)
                    else:
                        metrics[k] = [v]
            except:
                print(metrics, batch_idx, len(loader), len(loader))

            if rank == 0: #batch_idx % config.eval_vis_freq == 0 and rank == 0:
                vis_utils.vis_pred_clip(sample=sample,
                                        pred=preds_top,
                                        iter_num=batch_idx,
                                        output_dir=output_dir,
                                        subfolder='val')
                vis_utils.vis_pred_scores(sample=sample,
                                        pred=preds_top,
                                        iter_num=batch_idx,
                                        output_dir=output_dir,
                                        subfolder='val')
            dist.barrier()
    
    # print("metrics: ", metrics)
            
    # if rank == 0:
    #     wandb_log = {}
    #     for k in metrics.keys():
    #         wandb_log['Valid/{}'.format(k)] = torch.tensor(metrics[k]).mean().item()
    #     wandb_run.log(wandb_log)
    
    return torch.tensor(metrics['iou']).mean().item(), torch.tensor(metrics['prob_accuracy']).mean().item()


def val_performance(config, preds, gts, prob_theta=0.5):
    pred_center = preds['center']   # [b,t,N,2]
    pred_hw = preds['hw']           # [b,t,N,2], actually half of hw
    pred_bbox = preds['bbox']       # [b,t,N,4]
    pred_prob = preds['prob']       # [b,t,N]
    if 'prob_refine' in preds.keys():
        pred_prob_refine = preds['prob_refine']     # [b,t]
    b,t,N = pred_prob.shape

    pred_prob = rearrange(pred_prob, 'b t N -> (b t) N')
    pred_hw = rearrange(pred_hw, 'b t N c -> (b t) N c')
    pred_center = rearrange(pred_center, 'b t N c -> (b t) N c')
    pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t) N c')

    pred_prob, top_idx = torch.max(pred_prob, dim=-1)  # [b*t], [b*t]
    pred_bbox = torch.gather(pred_bbox, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,4)).squeeze()       # [b*t,4]
    pred_hw = torch.gather(pred_hw, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)).squeeze()           # [b*t,2]
    pred_center = torch.gather(pred_center, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,2)).squeeze()   # [b*t,2]

    if 'center' not in gts.keys():
        gts['center'] = (gts['clip_bbox'][...,:2] + gts['clip_bbox'][...,2:]) / 2.0
    if 'hw' not in gts.keys():
        gts['hw'] = gts['center'] - gts['clip_bbox'][...,:2]
    gt_center = rearrange(gts['center'], 'b t c -> (b t) c')
    gt_hw = rearrange(gts['hw'], 'b t c -> (b t) c')
    gt_bbox = rearrange(gts['clip_bbox'], 'b t c -> (b t) c')
    gt_prob = gts['clip_with_bbox'].reshape(-1)
    gt_before_query = gts['before_query'].reshape(-1)

    # bbox loss
    loss_center = F.l1_loss(pred_center[gt_prob.bool()], gt_center[gt_prob.bool()])
    loss_hw = F.l1_loss(pred_hw[gt_prob.bool()], gt_hw[gt_prob.bool()])
    iou_all, giou, loss_giou = GiouLoss(pred_bbox, gt_bbox, mask=gt_prob.bool())
    if torch.sum(gt_prob).item() > 0:
        iou = torch.mean(iou_all[gt_prob.bool()])
        iou_25 = torch.mean((iou_all[gt_prob.bool()] > 0.25).float())
        giou = torch.mean(giou[gt_prob.bool()])
    else:
        iou, iou_25, giou = 0.0, 0.0, 0.0
    
    # occurance loss
    weight = torch.tensor(config.loss.prob_bce_weight).to(gt_prob.device)
    weight_ = weight[gt_prob[gt_before_query.bool()].long()].reshape(-1)
    criterion = nn.BCEWithLogitsLoss(reduce=False)
    loss_prob = (criterion(pred_prob[gt_before_query.bool()].float(), 
                           gt_prob[gt_before_query.bool()].float()) * weight_).mean()
    
    if 'prob_refine' in preds.keys():
        pred_prob = pred_prob_refine.reshape(-1)
    
    prob_accuracy = ((torch.sigmoid(pred_prob) > prob_theta) == gt_prob.bool()).float().mean()
    prob_accuracy_2 = ((torch.sigmoid(pred_prob) > 0.6) == gt_prob.bool()).float().mean()
    prob_accuracy_3 = ((torch.sigmoid(pred_prob) > 0.7) == gt_prob.bool()).float().mean()
    prob_accuracy_4 = ((torch.sigmoid(pred_prob) > 0.65) == gt_prob.bool()).float().mean()
    
    loss = {
        # losses
        'loss_bbox_center': loss_center.item(),
        'loss_bbox_hw': loss_hw.item(),
        'loss_bbox_giou': loss_giou.item(),
        'loss_prob': loss_prob.item(),
        # information
        'iou': iou.item(),
        'iou_25': iou_25.item(),
        'giou': giou.item(),
        'prob_accuracy': prob_accuracy.item(),
        'prob_accuracy_0.6': prob_accuracy_2.item(),
        'prob_accuracy_0.7': prob_accuracy_3.item(),
        'prob_accuracy_0.65': prob_accuracy_4.item(),
    }

    # get top prediction
    pred_prob = rearrange(pred_prob, '(b t) -> b t', b=b, t=t)           # [b,t]
    pred_bbox = rearrange(pred_bbox, '(b t) c -> b t c', b=b, t=t)       # [b,t,4]
    pred_top ={
        'bbox': pred_bbox,
        'prob': pred_prob
    }

    return loss, pred_top