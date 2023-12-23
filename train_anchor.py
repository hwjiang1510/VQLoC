import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import itertools
import argparse

import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast

from config.config import config, update_config

from model.corr_clip_spatial_transformer2_anchor_2heads_hnm import ClipMatcher
from utils import exp_utils, train_utils, dist_utils
from dataset import dataset_utils
from func.train_anchor import train_epoch, validate

import transformers
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description='Train hand reconstruction network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        "--eval", dest="eval", action="store_true",help="evaluate model")
    parser.add_argument(
        '--local_rank', default=-1, type=int, help='node rank for distributed training')
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    # Get args and config
    args = parse_args()
    logger, output_dir, tb_log_dir = exp_utils.create_logger(config, args.cfg, phase='train')
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # set random seeds
    torch.cuda.manual_seed_all(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # set device
    gpus = range(torch.cuda.device_count())
    distributed = torch.cuda.device_count() > 1
    device = torch.device('cuda') if len(gpus) > 0 else torch.device('cpu')
    if "LOCAL_RANK" in os.environ:
        dist_utils.dist_init(int(os.environ["LOCAL_RANK"]))
    local_rank = dist.get_rank()
    torch.cuda.set_device(local_rank)

    # if local_rank == 0:
    #     wandb_name = config.exp_name
    #     wandb_proj_name = config.exp_group
    #     wandb_run = wandb.init(project=wandb_proj_name, group=wandb_name)#, name='smooth-puddle-94', resume=True)
    #     wandb.config.update({
    #         "exp_name": config.exp_name,
    #         "batch_size": config.train.batch_size,
    #         "total_iteration": config.train.total_iteration,
    #         "lr": config.train.lr,
    #         "weight_decay": config.train.weight_decay,
    #         "loss_weight_bbox_giou": config.loss.weight_bbox_giou,
    #         "loss_prob_bce_weight": config.loss.prob_bce_weight,
    #         "model_num_transformer": config.model.num_transformer,
    #         "model_resolution_transformer": config.model.resolution_transformer,
    #         "model_window_transformer": config.model.window_transformer,
    #     })
    # else:
    wandb_run = None

    # get model
    model = ClipMatcher(config).to(device)
    #model = torch.compile(model)

    # get optimizer
    optimizer = train_utils.get_optimizer(config, model)
    # schedular = train_utils.get_schedular(config, optimizer)
    schedular = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=config.train.schedular_warmup_iter,
                                                             num_training_steps=config.train.total_iteration)
    scaler = torch.cuda.amp.GradScaler()

    best_iou, best_prob = 0.0, 0.0
    ep_resume = None
    if config.train.resume:
        try:
            model, optimizer, schedular, scaler, ep_resume, best_iou, best_prob = train_utils.resume_training(
                                                                                model, optimizer, schedular, scaler, 
                                                                                output_dir,
                                                                                cpt_name='cpt_last.pth.tar')
            print('LR after resume {}'.format(optimizer.param_groups[0]['lr']))
        except:
            print('Resume failed')

    # distributed training
    ddp = False
    model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        device_num = len(device_ids)
        ddp = True

    # get dataset and dataloader    
    train_data = dataset_utils.get_dataset(config, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=config.train.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=True,
                                               sampler=train_sampler)
    val_data = dataset_utils.get_dataset(config, split='val')
    val_loader = torch.utils.data.DataLoader(val_data,
                                               batch_size=config.test.batch_size, 
                                               shuffle=False,
                                               num_workers=int(config.workers), 
                                               pin_memory=True, 
                                               drop_last=False)    
 
    start_ep = ep_resume if ep_resume is not None else 0
    end_ep = 100000000 #int(config.train.total_iteration / len(train_loader)) + 1

    # train
    for epoch in range(start_ep, end_ep):
        train_sampler.set_epoch(epoch)
        train_epoch(config,
                    loader=train_loader,
                    model=model,
                    optimizer=optimizer,
                    schedular=schedular,
                    scaler=scaler,
                    epoch=epoch,
                    output_dir=output_dir,
                    device=device,
                    rank=local_rank,
                    ddp=ddp,
                    wandb_run=wandb_run
                    )
        torch.cuda.empty_cache()

        if local_rank == 0:
            train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'schedular': schedular.state_dict(),
                        'scaler': scaler.state_dict(),
                    }, 
                    checkpoint=output_dir, filename="cpt_last.pth.tar")

        if epoch % 5 == 0:
            print('Doing validation...')
            iou, prob = validate(config,
                                loader=val_loader,
                                model=model,
                                epoch=epoch,
                                output_dir=output_dir,
                                device=device,
                                rank=local_rank,
                                ddp=ddp,
                                wandb_run=wandb_run
                                )
            torch.cuda.empty_cache()
            if iou > best_iou:
                best_iou = iou
                if local_rank == 0:
                    train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'schedular': schedular.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_iou': best_iou,
                    }, 
                    checkpoint=output_dir, filename="cpt_best_iou.pth.tar")

            if prob > best_prob:
                best_prob = prob
                if local_rank == 0:
                    train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'schedular': schedular.state_dict(),
                        'scaler': scaler.state_dict(),
                        'best_prob': best_prob,
                    }, 
                    checkpoint=output_dir, filename="cpt_best_prob.pth.tar")

            logger.info('Rank {}, best iou: {} (current {}), best probability accuracy: {} (current {})'.format(local_rank, best_iou, iou, best_prob, prob))
        dist.barrier()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()