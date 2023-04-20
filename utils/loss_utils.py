import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torchvision import transforms
from einops import rearrange
from model.corr_clip_anchor import default_aspect_ratios
from utils.anchor_utils import assign_labels


def get_losses_with_anchor(config, preds, gts):
    pred_center = preds['center']   # [b,t,N,2]
    pred_hw = preds['hw']           # [b,t,N,2], actually half of hw
    pred_bbox = preds['bbox']       # [b,t,N,4]
    pred_prob = preds['prob']       # [b,t,N]
    b,t,N = pred_prob.shape
    device = pred_prob.device

    if 'center' not in gts.keys():
        gts['center'] = (gts['clip_bbox'][...,:2] + gts['clip_bbox'][...,2:]) / 2.0
    if 'hw' not in gts.keys():
        gts['hw'] = gts['center'] - gts['clip_bbox'][...,:2]   # actually half of hw
    gt_center = gts['center']               # [b,t,2]
    gt_hw = gts['hw']                       # [b,t,2]
    gt_bbox = gts['clip_bbox']              # [b,t,4]
    gt_prob = gts['clip_with_bbox']         # [b,t]
    gt_before_query = gts['before_query']   # [b,t]

    assign_label = assign_labels(pred_bbox, gt_bbox, iou_threshold=config.model.positive_threshold)          # [b,t,N]
    positive = torch.logical_and(gt_prob.unsqueeze(-1).repeat(1,1,N).bool(),
                                 assign_label.bool())                            # [b,t,N]
    positive = rearrange(positive, 'b t N -> (b t N)')                           # [b*t*N]
    loss_mask = positive.float().unsqueeze(1)                                    # [b*t*N,1]
    
    if torch.sum(positive.float()).item() > 0:
        # bbox center loss
        pred_center = rearrange(pred_center, 'b t N c -> (b t N) c')
        pred_center_positive = pred_center[positive.bool()]
        gt_center_positive = rearrange(gt_center.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')[positive.bool()]
        loss_center = F.l1_loss(pred_center_positive, gt_center_positive)
        
        # bbox hw loss
        pred_hw = rearrange(pred_hw, 'b t N c -> (b t N) c')
        pred_hw_positive = pred_hw[positive.bool()]
        gt_hw_positive = rearrange(gt_hw.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')[positive.bool()]
        loss_hw = F.l1_loss(pred_hw_positive, gt_hw_positive)

        # bbox giou loss
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t N) c')
        gt_bbox_replicate = rearrange(gt_bbox.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')
        iou, giou, loss_giou = GiouLoss(pred_bbox, gt_bbox_replicate, mask=loss_mask.bool().squeeze())
    else:
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t N) c')
        loss_center = torch.tensor(0.).cuda()
        loss_hw = torch.tensor(0.).cuda()
        loss_giou = torch.tensor(0.).cuda()
        iou = torch.tensor(0.).cuda()
        giou = torch.tensor(0.).cuda()

    # occurance loss
    pred_prob = rearrange(pred_prob, 'b t N -> (b t N)')
    gt_before_query_replicate = rearrange(gt_before_query.unsqueeze(2).repeat(1,1,N), 'b t N -> (b t N)')
    # weight = torch.tensor(config.loss.prob_bce_weight).to(gt_prob.device)
    # weight_ = weight[positive[gt_before_query_replicate.bool()].long()].reshape(-1)
    # criterion = nn.BCEWithLogitsLoss(reduce=False)
    # loss_prob = (criterion(pred_prob[gt_before_query_replicate.bool()].float(),
    #                        positive[gt_before_query_replicate.bool()].float())
    #             * weight_).mean()

    # loss_prob using focal loss
    loss_prob = focal_loss(pred_prob[gt_before_query_replicate.bool()].float(),
                           positive[gt_before_query_replicate.bool()].float())

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
        # information
        'iou': iou.detach(),
        'giou': giou.detach()
    }

    # get top prediction
    pred_prob = rearrange(pred_prob, '(B N) -> B N', N=N)                                       # [b*t,N]
    pred_bbox = rearrange(pred_bbox, '(B N) c -> B N c', N=N)                                   # [b*t,N,4]
    pred_prob_top, top_idx = torch.max(pred_prob, dim=-1)                                       # [b*t], [b*t]
    pred_bbox_top = torch.gather(pred_bbox, dim=1, index=top_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,4)).squeeze()   # [b*t,4]
    pred_top ={
        'bbox': rearrange(pred_bbox_top, '(b t) c -> b t c', b=b, t=t),
        'prob': rearrange(pred_prob_top, '(b t) -> b t', b=b, t=t)
    }

    return loss, pred_top



def get_losses(config, preds, gts):
    pred_center = rearrange(preds['center'], 'b t c -> (b t) c')
    pred_hw = rearrange(preds['hw'], 'b t c -> (b t) c')
    pred_bbox = rearrange(preds['bbox'], 'b t c -> (b t) c')
    pred_prob = preds['prob'].reshape(-1)

    if 'center' not in gts.keys():
        gts['center'] = (gts['clip_bbox'][...,:2] + gts['clip_bbox'][...,2:]) / 2.0
    if 'hw' not in gts.keys():
        gts['hw'] = gts['center'] - gts['clip_bbox'][...,:2]   # actually half hw
    gt_center = rearrange(gts['center'], 'b t c -> (b t) c')
    gt_hw = rearrange(gts['hw'], 'b t c -> (b t) c')
    gt_bbox = rearrange(gts['clip_bbox'], 'b t c -> (b t) c')
    gt_prob = gts['clip_with_bbox'].reshape(-1)
    gt_before_query = gts['before_query'].reshape(-1)
    gt_ratio = get_bbox_ratio(gt_hw, gt_hw.device).reshape(-1)

    # bbox loss
    loss_center = F.l1_loss(pred_center[gt_prob.bool()], gt_center[gt_prob.bool()])
    loss_hw = F.l1_loss(pred_hw[gt_prob.bool()], gt_hw[gt_prob.bool()])
    #loss_bbox = F.l1_loss(pred_bbox[gt_prob.bool()], gt_bbox[gt_prob.bool()])
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
        #'loss_bbox': loss_bbox,
        'loss_bbox_giou': loss_giou,
        'loss_prob': loss_prob,
        # weights
        #'weight_bbox': config.loss.weight_bbox,
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
    :param mask: ground truth of valid instance, in shape [B]
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


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    '''
    focal loss for binary classification (background/foreground)
    inputs and targets in shape [N]
    inputs are not activated by sigmoid
    alpha is the weight for negatives (background)
    '''
    targets = targets.float()
    device = targets.device

    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    pt = torch.sigmoid(inputs)
    pt = torch.where(targets == 1, pt, 1 - pt)

    alpha = torch.where(targets == 1, 1 - alpha, alpha).to(device)

    F_loss = alpha * (1 - pt)**gamma * BCE_loss

    return F_loss.mean()

