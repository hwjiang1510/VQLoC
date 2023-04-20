import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from model.transformer import Block
#from model.local_transformer import Block_local
from utils.model_utils import PositionalEncoding1D, positionalencoding1d, positionalencoding3d, positionalencoding2d
from utils.model_utils import BasicBlock_Conv2D, BasicBlock_MLP
from einops import rearrange
import math


def build_backbone(config):
    name, type = config.model.backbone_name, config.model.backbone_type
    if name == 'dino':
        assert type in ['vitb8', 'vitb16', 'vits8', 'vits16']
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_{}'.format(type))
        down_rate = int(type.replace('vitb', '').replace('vits', ''))
        backbone_dim = 768
        if type == 'vitb16' and config.model.bakcbone_use_mae_weight:
            mae_weight = torch.load('/vision/hwjiang/episodic-memory/VQ2D/checkpoint/mae_pretrain_vit_base.pth')['model']
            backbone.load_state_dict(mae_weight)
    return backbone, down_rate, backbone_dim


class ClipMatcher(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # build backbone
        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)

        # model config based on backbone
        self.query_size = config.dataset.query_size
        self.clip_size_fine = config.dataset.clip_size_fine
        self.clip_size_coarse = config.dataset.clip_size_coarse
        self.query_feat_size = self.query_size // self.down_rate
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate

        # spatial-temporal transformer config
        self.type_transformer = config.model.type_transformer
        self.window_transformer = config.model.window_transformer
        self.resolution_transformer = config.model.resolution_transformer

        # other model config
        self.type_pe2d = 'zero'
        self.clip_dim = 256
        self.query_dim = 256
        self.feat_dim = self.clip_dim + self.query_dim

        '''Coarse module parameters'''
        self.clip_reduce = BasicBlock_Conv2D(self.backbone_dim, self.clip_dim)

        # query shape in [(1,1), (2,2), (4,4), ..., (query_feat_size//2, query_feat_size//2)]
        self.query_target_size, self.down_query_heads = [], []
        for i in range(int(math.log2(self.query_feat_size))):
            self.query_target_size.append((2**i, 2**i))              
            self.down_query_heads.append(BasicBlock_Conv2D(self.backbone_dim, self.query_dim))
        self.down_query_heads = nn.ModuleList(self.down_query_heads)

        self.pe_2d = positionalencoding2d(self.feat_dim, self.clip_feat_size_coarse, self.clip_feat_size_coarse, self.type_pe2d)
        self.pe_2d = nn.parameter.Parameter(0.1 * rearrange(self.pe_2d, '(h w) c -> c h w', h=self.clip_feat_size_coarse)).unsqueeze(0)

        self.num_head_layers = int(math.log2(self.clip_feat_size_coarse))
        self.down_heads = nn.ModuleList([BasicBlock_Conv2D(self.feat_dim, self.feat_dim) for _ in range(self.num_head_layers)])

        self.pe_3d = positionalencoding3d(d_model=self.feat_dim, 
                                          height=self.resolution_transformer, 
                                          width=self.resolution_transformer, 
                                          depth=config.dataset.clip_num_frames,
                                          type=config.model.pe_transformer).unsqueeze(0)
        self.pe_3d = nn.parameter.Parameter(self.pe_3d)

        self.transformer_layer = []
        self.num_transformer = config.model.num_transformer
        for _ in range(self.num_transformer):
            self.transformer_layer.append(
                torch.nn.TransformerEncoderLayer(
                    d_model=self.feat_dim, 
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True))
        self.transformer_layer = nn.ModuleList(self.transformer_layer)
        self.temporal_mask = None

        self.out = BasicBlock_MLP([self.feat_dim, 256, 5])
        self.out.apply(self.init_weights_linear)

        '''Fine module parameters'''
        self.fine_down_head = BasicBlock_Conv2D(self.backbone_dim*2, self.feat_dim)
        self.down_heads_fine = nn.ModuleList([BasicBlock_Conv2D(self.feat_dim, self.feat_dim) for _ in range(self.num_head_layers)])
        self.out_fine = BasicBlock_MLP([self.feat_dim, 256, 4])
        self.out_fine.apply(self.init_weights_linear)


    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)


    def extract_feature(self, x, return_h_w=False):
        b, _, h_origin, w_origin = x.shape
        out = self.backbone.get_intermediate_layers(x, n=1)[0]
        out = out[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
        h, w = int(h_origin / self.backbone.patch_embed.patch_size), int(w_origin / self.backbone.patch_embed.patch_size)
        dim = out.shape[-1]
        out = out.reshape(b, h, w, dim).permute(0,3,1,2)
        if return_h_w:
            return out, h, w
        return out


    def forward_coarse(self, clip, query, fix_backbone=True, return_fine=False):
        '''
        clip: in shape [b,t,c,h,w]
        query: in shape [b,c,h2,w2]
        '''
        b, t, c, h, w = clip.shape
        h2, w2 = query.shape[-2:]
        clip = rearrange(clip, 'b t c h w -> (b t) c h w')

        # get backbone features
        if fix_backbone:
            with torch.no_grad():
                query_feat = self.extract_feature(query)
                clip_feat = self.extract_feature(clip)
        else:
            query_feat = self.extract_feature(query)
            clip_feat = self.extract_feature(clip)

        query_feat_origin = query_feat.clone()
        clip_feat_origin = clip_feat.clone()  

        # resize backbone features
        clip_feat = self.clip_reduce(clip_feat)
        if [clip_feat.shape[-2], clip_feat.shape[-1]] != [self.clip_feat_size_coarse, self.clip_feat_size_coarse]:
            clip_feat = F.interpolate(clip_feat, (self.clip_feat_size_coarse, self.clip_feat_size_coarse), mode='bilinear')

        # make multiscale query template
        query_down = torch.zeros(b, 256, self.clip_feat_size_coarse, self.clip_feat_size_coarse).to(clip_feat.device)
        for size, down_head in zip(self.query_target_size, self.down_query_heads):
            cur_query = down_head(F.interpolate(query_feat, size, mode='bilinear'))
            repeat = (1, 1, self.clip_feat_size_coarse // size[0], self.clip_feat_size_coarse // size[1])
            cur_query = cur_query.repeat(repeat)
            query_down += cur_query
        query_down = query_down.unsqueeze(1).repeat(1,t,1,1,1)
        query_feat = rearrange(query_down, 'b t c h w -> (b t) c h w')

        # downsample the per-frame features, use transformer for leveraging temporal information
        feat = torch.cat([clip_feat, query_feat], dim=1) + self.pe_2d   
        for cur_head in self.down_heads:
            cur_shape = feat.shape[2:]
            feat = cur_head(F.interpolate(feat, (cur_shape[0]//2, cur_shape[1]//2), mode='bilinear'))
        token = feat.squeeze()      # [b*t,512]

        # make initial prediction
        pred = self.out(token)
        pred = rearrange(pred, '(b t) c -> b t c', b=b, t=t)
        center, hw, prob = pred.split([2, 2, 1], dim=-1)
        bbox = torch.cat([center - hw, center + hw], dim=-1)

        result = {
            'center': center,               # [b,t,2]
            'hw': hw,                       # [b,t,2]
            'bbox': bbox,                   # [b,t,4]
            'prob': prob.squeeze(-1)        # [b,t]
        }
        if return_fine:
            result['clip_feat_origin'] = clip_feat_origin
            result['query_feat_origin'] = query_feat_origin
        return result
    

    def forward_fine(self, result_corase):
        bbox = result['bbox']
        scale = (bbox[..., 2:] - bbox[...,:2]).repeat(1,1,2)    # [b,t,4]
        clip_feat_origin = result['clip_feat_origin']
        query_feat_origin = result['query_feat_origin']

        # roialign by initial prediction
        bbox_roi = rearrange(bbox, 'b t c -> (b t) c') * self.clip_feat_size_coarse
        clip_feat_crop = torchvision.ops.roi_align(clip_feat_origin, boxes=bbox_roi, output_size=self.clip_feat_size_coarse)
        
        # make refinement
        clip_feat_crop = self.fine_down_head(torch.cat([clip_feat_crop, query_feat_origin], dim=1))
        for cur_head in self.down_heads_fine:
            cur_shape = clip_feat_crop.shape[2:]
            clip_feat_crop = cur_head(F.interpolate(clip_feat_crop, (cur_shape[0]//2, cur_shape[1]//2), mode='bilinear'))
        token = clip_feat_crop.squeeze()
        bbox_delta = self.out_fine(token)
        bbox_delta = rearrange(bbox_delta, '(b t) c -> b t c', b=bbox.shape[0]) * scale
        bbox_fine = bbox + bbox_delta

        center = (bbox_fine[...,:2] + bbox_fine[...,:2]) / 2.0
        hw = center - bbox_fine[...,:2]
        
        result = {
            'center': center,
            'hw': hw,
            'bbox': bbox_fine,
            'prob': result_corase['prob']
        }
        return result
    

    def forward(self, clip, query, fix_backbone=True, return_fine=False):
        '''
        clip: in shape [b,t,c,h,w]
        query: in shape [b,c,h2,w2]
        '''
        b, t, c, h, w = clip.shape
        h2, w2 = query.shape[-2:]
        
        if not return_fine:
            result = self.forward_coarse(clip, query, fix_backbone, return_fine)
            return result
        else:
            with torch.no_grad():
                result_coarse = self.forward_coarse(clip, query, fix_backbone, return_fine)
            result_fine = self.forward_fine(result_coarse)
            return result_fine
        
    
    def get_mask(self, src, t):
        if not torch.is_tensor(self.temporal_mask):
            hw = src.shape[1] // t
            thw = src.shape[1]
            mask = torch.zeros(thw, thw)

            window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, (i-window_size)*(hw))
                max_idx = min(thw, (i+window_size+1)*(hw))
                mask[i*hw: i*(hw+1), min_idx: max_idx] = float('-inf')
            mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask