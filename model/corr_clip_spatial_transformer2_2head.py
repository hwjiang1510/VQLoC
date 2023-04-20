import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import Block
#from model.local_transformer import Block_local
from utils.model_utils import PositionalEncoding1D, positionalencoding1d, positionalencoding3d, positionalencoding2d
from einops import rearrange
import math
from utils.model_utils import BasicBlock_Conv2D, BasicBlock_MLP


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
    elif name == 'dinov2':
        assert type in ['vits14', 'vitb14', 'vitl14', 'vitg14']
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_{}'.format(type))
        down_rate = 14
        backbone_dim = 768
    return backbone, down_rate, backbone_dim


class ClipMatcher(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.backbone, self.down_rate, self.backbone_dim = build_backbone(config)
        self.backbone_name = config.model.backbone_name

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

        # query down heads
        self.query_down_heads = []
        for _ in range(int(math.log2(self.query_feat_size))):
            self.query_down_heads.append(
                nn.Sequential(
                    nn.Conv2d(self.backbone_dim, self.backbone_dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(self.backbone_dim),
                    nn.LeakyReLU(inplace=True),
                )
            )
        self.query_down_heads = nn.ModuleList(self.query_down_heads)

        # feature reduce layer
        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        
        # clip-query correspondence
        self.CQ_corr_transformer = []
        for _ in range(1):
            self.CQ_corr_transformer.append(
                torch.nn.TransformerDecoderLayer(
                    d_model=256,
                    nhead=4,
                    dim_feedforward=1024,
                    dropout=0.0,
                    activation='gelu',
                    batch_first=True
                )
            )
        self.CQ_corr_transformer = nn.ModuleList(self.CQ_corr_transformer)

        # feature downsample layers
        self.num_head_layers, self.down_heads = int(math.log2(self.clip_feat_size_coarse)), []
        for i in range(self.num_head_layers-1):
            self.in_channel = 256 if i != 0 else self.backbone_dim
            self.down_heads.append(
                nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ))
        self.down_heads = nn.ModuleList(self.down_heads)

        # spatial-temporal PE
        self.pe_3d = positionalencoding3d(d_model=256, 
                                          height=self.resolution_transformer, 
                                          width=self.resolution_transformer, 
                                          depth=config.dataset.clip_num_frames,
                                          type=config.model.pe_transformer).unsqueeze(0)
        self.pe_3d = nn.parameter.Parameter(self.pe_3d)

        # spatial-temporal transformer layer
        self.feat_corr_transformer = []
        self.num_transformer = config.model.num_transformer
        for _ in range(self.num_transformer):
            self.feat_corr_transformer.append(
                    torch.nn.TransformerEncoderLayer(
                        d_model=256, 
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True
                ))
        self.feat_corr_transformer = nn.ModuleList(self.feat_corr_transformer)
        self.temporal_mask = None

        # output head
        self.head = Head(in_dim=256)

    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def extract_feature(self, x, return_h_w=False):
        if self.backbone_name == 'dino':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            out = out[:, 1:, :]  # we discard the [CLS] token   # [b, h*w, c]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size), int(w_origin / self.backbone.patch_embed.patch_size)
            dim = out.shape[-1]
            out = out.reshape(b, h, w, dim).permute(0,3,1,2)
            if return_h_w:
                return out, h, w
            return out
        elif self.backbone_name == 'dinov2':
            b, _, h_origin, w_origin = x.shape
            out = self.backbone.get_intermediate_layers(x, n=1)[0]
            h, w = int(h_origin / self.backbone.patch_embed.patch_size[0]), int(w_origin / self.backbone.patch_embed.patch_size[1])
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
        b, t = clip.shape[:2]
        clip = rearrange(clip, 'b t c h w -> (b t) c h w')

        # get backbone features
        if fix_backbone:
            with torch.no_grad():
                query_feat = self.extract_feature(query)
                clip_feat = self.extract_feature(clip)
        else:
            query_feat = self.extract_feature(query)        # [b c h w]
            clip_feat = self.extract_feature(clip)          # (b t) c h w
        h, w = clip_feat.shape[-2:]

        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)
        
        query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')  # [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')                                         # [b*t,n,c]
        for layer in self.CQ_corr_transformer:
            clip_feat = layer(clip_feat, query_feat)                                                     # [b*t,n,c]
        
        clip_feat = rearrange(clip_feat, 'b (h w) c -> b c h w', h=h, w=w)
        #clip_feat = self.reduce(clip_feat)                                                               # [b*t,c,h,w]

        for head in self.down_heads:
            clip_feat = head(clip_feat)
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
                clip_feat = rearrange(clip_feat, '(b t) c h w ->b (t h w) c', b=b) + self.pe_3d
                mask = self.get_mask(clip_feat, t)
                for layer in self.feat_corr_transformer:
                    clip_feat = layer(clip_feat, src_mask=mask)
                clip_feat = rearrange(clip_feat, 'b (t h w) c -> (b t) c h w', b=b, t=t, h=self.resolution_transformer, w=self.resolution_transformer)
                break

        center, hw, bbox, prob = self.head(clip_feat, b=b, t=t)
        
        result = {
            'center': center,
            'hw': hw,
            'bbox': bbox,
            'prob': prob.squeeze(-1)
        }
        return result
    
    def get_mask(self, src, t):
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
    


class Head(nn.Module):
    def __init__(self, in_dim=256):
        super(Head, self).__init__()

        self.in_dim = in_dim
        self.num_layers = 2

        self.in_conv = BasicBlock_Conv2D(in_dim=in_dim, out_dim=2*in_dim)

        self.regression_conv = []
        for i in range(self.num_layers):
            self.regression_conv.append(
                nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ))
        self.regression_conv = nn.Sequential(*self.regression_conv)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(
                nn.Sequential(
                nn.Conv2d(256, 256, 3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(inplace=True),
            ))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        # self.regression_head = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.Dropout(p=0.2),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(256, 4),
        # )

        # self.classification_head = nn.Sequential(
        #     nn.Linear(1024, 512),
        #     nn.Dropout(p=0.2),
        #     nn.BatchNorm1d(512),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(256, 1),
        # )

        # self.regression_head.apply(self.init_weights_linear)
        # self.classification_head.apply(self.init_weights_linear)

        self.regression_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 4),
        )
        self.classification_head =  nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self.droupout_feat = torch.nn.Dropout(p=0.2)
        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.regression_head[-1].apply(self.init_weights_linear)
        self.classification_head[-1].apply(self.init_weights_linear)

    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x, b, t):
        '''
        x in shape [B, c, h, w]
        '''
        B, c, h, w = x.shape

        feat_reg, feat_cls = self.in_conv(x).split([c, c], dim=1)   # both [B,c,h,w]

        feat_reg = self.regression_conv(feat_reg).squeeze()         # [B,c]
        feat_cls = self.classification_conv(feat_cls).squeeze()     # [B,c]

        # feat_reg = self.regression_conv(feat_reg).squeeze()         # [B,c,2,2]
        # feat_cls = self.classification_conv(feat_cls).squeeze()     # [B,c,2,2]
        # feat_reg = rearrange(feat_reg, 'B c h w -> B (c h w)')      # [B,c]
        # feat_cls = rearrange(feat_cls, 'B c h w -> B (c h w)')      # [B,c]

        feat_reg = self.droupout_feat(feat_reg)
        feat_cls = self.droupout_cls(feat_cls)

        out_reg = self.regression_head(feat_reg)                    # [B,4]
        out_cls = self.classification_head(feat_cls)                # [B,1]

        out_reg = rearrange(out_reg, '(b t) c -> b t c', b=b, t=t)
        out_cls = rearrange(out_cls, '(b t) c -> b t c', b=b, t=t)

        center, hw = out_reg.split([2,2], dim=-1)
        bbox = torch.cat([center - hw, center + hw], dim=-1)

        return center, hw, bbox, out_cls
   