import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchvision import transforms
from einops import rearrange
from model.corr_clip_anchor import default_aspect_ratios


def get_losses(config, preds, gts):
    pred_center = rearrange(preds['center'], 'b t c -> (b t) c')
    pred_hw = rearrange(preds['hw'], 'b t c -> (b t) c')
    pred_bbox = rearrange(preds['bbox'], 'b t c -> (b t) c')
    pred_prob = preds['prob'].reshape(-1)

    if 'center' not in gts.keys():
        gts['center'] = (gts['clip_bbox'][...,:2] + gts['clip_bbox'][...,2:]) / 2.0
    if 'hw' not in gts.keys():
        gts['hw'] = gts['center'] - gts['clip_bbox'][...,:2]
    gt_center = rearrange(gts['center'], 'b t c -> (b t) c')
    gt_hw = rearrange(gts['hw'], 'b t c -> (b t) c')
    gt_bbox = rearrange(gts['clip_bbox'], 'b t c -> (b t) c')
    gt_prob = gts['clip_with_bbox'].reshape(-1)
    gt_before_query = gts['before_query'].reshape(-1)
    gt_ratio = get_bbox_ratio(gt_hw, gt_hw.device).reshape(-1)

    # bbox loss
    loss_center = F.l1_loss(pred_center[gt_prob.bool()], gt_center[gt_prob.bool()])
    loss_hw = F.l1_loss(pred_hw[gt_prob.bool()], gt_hw[gt_prob.bool()])
    iou, giou, loss_giou = GiouLoss(pred_bbox, gt_bbox, mask=gt_prob.bool())
    if 'bbox_ratio' in preds.keys():
        pred_ratio = preds['bbox_ratio'].reshape(-1)
        loss_ratio = F.l1_loss(pred_ratio[gt_prob.bool()], gt_ratio[gt_prob.bool()])
    
    # occurance loss
    weight = torch.tensor(config.loss.prob_bce_weight).to(gt_prob.device)
    weight_ = weight[gt_prob[gt_before_query.bool()].long()].reshape(-1)
    criterion = nn.BCEWithLogitsLoss(reduce=False)
    loss_prob = (criterion(pred_prob[gt_before_query.bool()], gt_prob[gt_before_query.bool()]) * weight_).mean()
    #loss_prob = F.binary_cross_entropy(pred_prob, gt_prob)
    loss = {
        'loss_bbox_center': loss_center,
        'loss_bbox_hw': loss_hw,
        'loss_bbox_giou': loss_giou,
        'loss_prob': loss_prob,
        # weights
        'weight_bbox_center': config.loss.weight_bbox_center,
        'weight_bbox_hw': config.loss.weight_bbox_hw,
        'weight_bbox_giou': config.loss.weight_bbox_giou,
        'weight_prob': config.loss.weight_prob,
        # informations
        'iou': iou.detach(),
        'giou': giou.detach()
    }
    if 'bbox_ratio' in preds.keys():
        loss.update({
                'loss_bbox_ratio': loss_ratio,
                'weight_bbox_ratio': config.loss.weight_bbox_ratio
            })
    return loss


def GiouLoss(bbox_p, bbox_g, mask):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :return:
    """
    device= bbox_p.device
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    x1p = torch.minimum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    x2p = torch.maximum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    y1p = torch.minimum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
    y2p = torch.maximum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)

    bbox_p = torch.cat([x1p, y1p, x2p, y2p], axis=1)
    # calc area of Bg
    area_p = (bbox_p[:, 2] - bbox_p[:, 0]) * (bbox_p[:, 3] - bbox_p[:, 1])
    # calc area of Bp
    area_g = (bbox_g[:, 2] - bbox_g[:, 0]) * (bbox_g[:, 3] - bbox_g[:, 1])

    # cal intersection
    x1I = torch.maximum(bbox_p[:, 0], bbox_g[:, 0])
    y1I = torch.maximum(bbox_p[:, 1], bbox_g[:, 1])
    x2I = torch.minimum(bbox_p[:, 2], bbox_g[:, 2])
    y2I = torch.minimum(bbox_p[:, 3], bbox_g[:, 3])
    I = torch.maximum((y2I - y1I), torch.tensor([0.0]).to(device)) * torch.maximum((x2I - x1I), torch.tensor([0.0]).to(device))

    # find enclosing box
    x1C = torch.minimum(bbox_p[:, 0], bbox_g[:, 0])
    y1C = torch.minimum(bbox_p[:, 1], bbox_g[:, 1])
    x2C = torch.maximum(bbox_p[:, 2], bbox_g[:, 2])
    y2C = torch.maximum(bbox_p[:, 3], bbox_g[:, 3])

    # calc area of Bc
    area_c = (x2C - x1C) * (y2C - y1C)
    U = area_p + area_g - I
    iou = 1.0 * I / (U + 1e-6)

    # Giou
    giou = iou - (area_c - U) / area_c
  
    loss_giou = torch.mean(1.0 - giou[mask])
    return iou, giou, loss_giou


def get_bbox_ratio(hw, device):
    '''
    params:
        hw: height and width of bbox, in shape [B,2]
    return:
        ratio: closest bbox aspect ratio in default_aspect_ratios, in shape [B]
    '''
    b = hw.shape[0]
    default_ratios = default_aspect_ratios.to(device)

    h, w = hw.split([1,1], dim=-1)
    ratio = h / w
    distance = torch.abs(ratio.repeat(1, default_ratios.shape[0]) - default_ratios.unsqueeze(0))     # [b,n]
    idx = torch.argmax(distance, dim=-1)    # [b]
    ratio_quant = torch.tensor([default_ratios[it.long()] for it in idx]).to(device)
    return ratio_quant



    