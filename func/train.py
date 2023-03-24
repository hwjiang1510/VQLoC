import torch
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

logger = logging.getLogger(__name__)


def train_epoch(config, loader, model, optimizer, schedular, scaler, epoch, output_dir, device, rank, ddp=True):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()
    
    train_utils.set_model_train(config, model, ddp)

    batch_end = time.time()
    for batch_idx, sample in enumerate(loader):
        iter_num = batch_idx + len(loader) * epoch

        sample = exp_utils.dict_to_cuda(sample)
        sample = dataset_utils.process_data(config, sample, split='train', device=device)     # normalize and data augmentations on GPU
        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        # for k, it in sample.items():
        #     print(k, it.shape)

        # reconstruction loss
        clips, queries = sample['clip'], sample['query']
        #with autocast():
        preds = model(clips, queries, fix_backbone=config.model.fix_backbone)
        time_meters.add_loss_value('Prediction time', time.time() - end)
        end = time.time()

        losses = loss_utils.get_losses(config, preds, sample)
        total_loss = 0.0
        for k, v in losses.items():
            if 'loss' in k:
                total_loss += losses[k.replace('loss_', 'weight_')] * v
                loss_meters.add_loss_value(k, v)
        total_loss = total_loss / config.train.accumulation_step
        dist.barrier()
            
        total_loss.backward()
        if (batch_idx+1) % config.train.accumulation_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()
            schedular.step()
        
        # #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
        # scaler.scale(total_loss).backward()
        # # scaler.unscale_(optimizer)
        # # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
        # scaler.step(optimizer)
        # scaler.update()
        # optimizer.zero_grad()
        # schedular.step()
        
        time_meters.add_loss_value('Batch time', time.time() - batch_end)

        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, all {batch_time:.3f}s, Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].val,
                recon_time=time_meters.average_meters['Prediction time'].val,
                batch_time=time_meters.average_meters['Batch time'].val
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.6f} ({loss.avg:.6f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)

        if iter_num % config.vis_freq == 0 and rank == 0:
            vis_utils.vis_pred_clip(sample=sample,
                                    pred=preds,
                                    iter_num=iter_num,
                                    output_dir=output_dir,
                                    subfolder='train')
        batch_end = time.time()
