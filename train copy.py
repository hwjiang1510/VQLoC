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

#from model.corr_clip import ClipMatcher
# from model.corr_clip_anchor import ClipMatcher
from model.corr_clip_spatial_transformer import ClipMatcher
from utils import exp_utils, train_utils
from dataset import dataset_utils
from func.train import train_epoch


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
    if device == torch.device("cuda"):
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        print('rank', args.local_rank)

    # get model
    model = ClipMatcher(config).to(device)
    model = torch.compile(model)

    # get optimizer
    optimizer = train_utils.get_optimizer(config, model)
    schedular = train_utils.get_schedular(config, optimizer)
    scaler = torch.cuda.amp.GradScaler()

    best_iou, best_prob = 0.0, float('inf')
    ep_resume = None
    if config.train.resume:
        model, optimizer, schedular, scaler, ep_resume, best_iou, best_prob = train_utils.resume_training(
                                                                            model, optimizer, schedular, scaler, 
                                                                            output_dir,
                                                                            cpt_name='cpt_last.pth.tar')

    # distributed training
    ddp = False
    model =  torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if device == torch.device("cuda"):
        torch.backends.cudnn.benchmark = True
        device_ids = range(torch.cuda.device_count())
        print("using {} cuda".format(len(device_ids)))
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
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
    end_ep = int(config.train.total_iteration / len(train_loader)) + 1

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
                    rank=args.local_rank,
                    ddp=ddp
                    )

        if args.local_rank == 0:
            train_utils.save_checkpoint(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'schedular': schedular.state_dict(),
                        'scaler': scaler.state_dict()
                    }, 
                    checkpoint=output_dir, filename="cpt_last.pth.tar")

        # if epoch % (1 * config.train.batch_size) == 0:#and args.local_rank == 0:
        #     print('Testing..')
            # cur_psnr, return_dict = test_nvs(config, 
            #         loader=val_loader,
            #         dataset=val_data,
            #         model=model,
            #         epoch=epoch, 
            #         output_dir=output_dir, 
            #         device=device,
            #         rank=args.local_rank)
            # torch.cuda.empty_cache()
            
            # if cur_psnr > best_psnr:
            #     best_psnr = cur_psnr
            #     if args.local_rank == 0:
            #         train_utils.save_checkpoint(
            #         {
            #             'epoch': epoch + 1,
            #             'state_dict': model.module.state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'best_psnr': best_psnr,
            #             'eval_dict': return_dict,
            #         }, 
            #         checkpoint=output_dir, filename="cpt_best_psnr_{}.pth.tar".format(best_psnr))
            
            # if args.local_rank == 0:
            #     logger.info('Best iou: {}, best probability loss: {}'.format(best_psnr))


if __name__ == '__main__':
    main()