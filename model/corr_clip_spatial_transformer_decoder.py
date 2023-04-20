import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import Block
#from model.local_transformer import Block_local
from utils.model_utils import PositionalEncoding1D, positionalencoding1d, positionalencoding3d
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
        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)

        self.query_size = config.dataset.query_size
        self.clip_size_fine = config.dataset.clip_size_fine
        self.clip_size_coarse = config.dataset.clip_size_coarse

        self.query_feat_size = self.query_size // self.down_rate
        self.clip_feat_size_fine = self.clip_size_fine // self.down_rate
        self.clip_feat_size_coarse = self.clip_size_coarse // self.down_rate

        self.type_transformer = config.model.type_transformer
        assert self.type_transformer in ['local', 'global']
        self.window_transformer = config.model.window_transformer
        self.resolution_transformer = config.model.resolution_transformer

        self.clip_reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.query_target_size, self.down_query_heads = [], []
        for i in range(int(math.log2(self.query_feat_size))):
            self.query_target_size.append((2**i, 2**i))                 # [(1,1), (2,2), (4,4), ..., (query_feat_size//2, query_feat_size//2)]
            self.down_query_heads.append(nn.Sequential(
                    nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(inplace=True),
                ))
        self.down_query_heads = nn.ModuleList(self.down_query_heads)

        self.feat_process = nn.Sequential(
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(inplace=True),
                )

        self.num_head_layers, self.down_heads = int(math.log2(self.clip_feat_size_coarse)), []
        for _ in range(self.num_head_layers):
            self.down_heads.append(
                nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(inplace=True),
            ))
        self.down_heads = nn.ModuleList(self.down_heads)

        self.pe_3d = positionalencoding3d(d_model=512, 
                                          height=self.resolution_transformer, 
                                          width=self.resolution_transformer, 
                                          depth=config.dataset.clip_num_frames,
                                          type=config.model.pe_transformer).unsqueeze(0)#.permute(0,2,1) #PositionalEncoding1D(channels=512)
        self.pe_3d = nn.parameter.Parameter(self.pe_3d)

        self.transformer_layer = []
        self.num_transformer = config.model.num_transformer
        for _ in range(self.num_transformer):
            if self.type_transformer == 'global':
                self.transformer_layer.append(
                    torch.nn.TransformerDecoderLayer(
                        d_model=512, 
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True
                ))
        self.transformer_layer = nn.ModuleList(self.transformer_layer)
        self.temporal_mask = None
        self.feat_spatial_mask = None

        self.out = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 5)
        )
        self.out.apply(self.init_weights_linear)


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


    def forward(self, clip, query, fix_backbone=True):
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

        # resize backbone features
        #feat_spatial = rearrange(self.feat_process(clip_feat), '(b t) c h w -> b (t h w) c', b=b)
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
        feat = torch.cat([clip_feat, query_feat], dim=1)
        feat_spatial = rearrange(self.feat_process(feat), '(b t) c h w -> b (t h w) c', b=b)      
        for cur_head in self.down_heads:
            cur_shape = feat.shape[2:]
            feat = cur_head(F.interpolate(feat, (cur_shape[0]//2, cur_shape[1]//2), mode='bilinear'))
            if list(feat.shape[-2:]) == [self.resolution_transformer, self.resolution_transformer]:
                feat = rearrange(feat, '(b t) c h w -> b (t h w) c', b=b) + self.pe_3d
                feat_spatial_mask = self.get_mask(feat, feat_spatial, t)
                temporal_mask = self.get_mask_temporal(feat, t)
                for layer in self.transformer_layer:    # cross transformer
                    feat = layer(feat, feat_spatial, tgt_mask=temporal_mask, memory_mask=feat_spatial_mask)
                feat = rearrange(feat, 'b (t h w) c -> (b t) c h w', b=b, h=self.resolution_transformer, w=self.resolution_transformer)
        token = feat.squeeze()      # [b*t,512]

        # make prediction
        pred = self.out(token)
        pred = rearrange(pred, '(b t) c -> b t c', b=b, t=t)
        center, hw, prob = pred.split([2, 2, 1], dim=-1)
        bbox = torch.cat([center - hw, center + hw], dim=-1)
        
        result = {
            'center': center,
            'hw': hw,
            'bbox': bbox,
            'prob': prob.squeeze(-1)
        }
        return result
    

    def get_mask(self, token, feat, t):
        '''
        token: in shape [b, s*t, c], s is the spatial resolution of token from one image, t is time
                in most cases s=1*1
        feat: in shape [b, h*w*t, c]
        '''
        if not torch.is_tensor(self.feat_spatial_mask):
            b = token.shape[0]
            s = token.shape[1] // t
            hw = feat.shape[1] // t
            device = token.device

            mask = torch.ones(token.shape[1], feat.shape[1]) * float('-inf')
            for i in range(t):
                mask[i*s: (i+1)*s, i*hw: (i+1)*hw] = 0.0
            mask = mask.to(device)
            self.feat_spatial_mask = mask
        return self.feat_spatial_mask


    def get_mask_temporal(self, src, t):
        if not torch.is_tensor(self.temporal_mask):
            hw = src.shape[1] // t
            thw = src.shape[1]
            mask = torch.ones(thw, thw).float() * float('-inf')

            window_size = self.window_transformer // 2

            for i in range(t):
                min_idx = max(0, (i-window_size)*hw)
                max_idx = min(thw, (i+window_size+1)*hw)
                mask[i*hw: (i+1)*hw, min_idx: max_idx] = 0.0
            mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask


