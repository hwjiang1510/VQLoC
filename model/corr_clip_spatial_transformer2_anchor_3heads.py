import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import Block
#from model.local_transformer import Block_local
from utils.model_utils import PositionalEncoding1D, positionalencoding1d, positionalencoding3d, positionalencoding2d
from utils.model_utils import BasicBlock_Conv2D, BasicBlock_MLP
from utils.anchor_utils import generate_anchor_boxes_on_regions
from einops import rearrange
import math


base_sizes=torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32)    # 4 types of size
aspect_ratios=torch.tensor([0.5, 1, 2], dtype=torch.float32)                                # 3 types of aspect ratio
n_base_sizes = base_sizes.shape[0]
n_aspect_ratios = aspect_ratios.shape[0]


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

        self.anchors = generate_anchor_boxes_on_regions(image_size=[self.clip_size_coarse, self.clip_size_coarse],
                                                        num_regions=[self.resolution_transformer, self.resolution_transformer])
        self.anchors = self.anchors.unsqueeze(0) / self.clip_size_coarse   # [1,R^2*N*M,4], value range [0,1], represented by [c_x,c_y,h,w] in torch axis

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

        # probability refinement head
        self.head_prob = Head_prob(config=config,
                                   h=self.resolution_transformer, 
                                   w=self.resolution_transformer,
                                   dim=n_aspect_ratios*n_base_sizes)

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
        
        clip_feat = rearrange(clip_feat, 'b (h w) c -> b c h w', h=h, w=w)                               # [b*t,c,h,w]

        for head in self.down_heads:
            clip_feat = head(clip_feat)
            if list(clip_feat.shape[-2:]) == [self.resolution_transformer]*2:
                clip_feat = rearrange(clip_feat, '(b t) c h w ->b (t h w) c', b=b) + self.pe_3d
                mask = self.get_mask(clip_feat, t)
                for layer in self.feat_corr_transformer:
                    clip_feat = layer(clip_feat, src_mask=mask)
                clip_feat = rearrange(clip_feat, 'b (t h w) c -> (b t) c h w', b=b, t=t, h=self.resolution_transformer, w=self.resolution_transformer)
                break
        
        bbox_refine, prob = self.head(clip_feat)                                # [b*t,N=h*w*n*m,c]
        bbox_refine = rearrange(bbox_refine, '(b t) N c -> b t N c', b=b, t=t)  # [b,t,N,4]
        prob = rearrange(prob, '(b t) N c -> b t N c', b=b, t=t)                # [b,t,N,1]
        bbox_refine += self.anchors.to(bbox_refine.device)
        center, hw = bbox_refine.split([2,2], dim=-1)                           # represented by [c_x, c_y, h, w]
        hw = 0.5 * hw                                                           # anchor's hw is defined as real hw
        bbox = torch.cat([center - hw, center + hw], dim=-1)                    # [b,t,N,4]
        
        prob_refine = self.head_prob(prob)                                # [b,t]

        result = {
            'center': center,           # [b,t,N,2]
            'hw': hw,                   # [b,t,N,2]
            'bbox': bbox,               # [b,t,N,4]
            'prob': prob.squeeze(-1),   # [b,t,N]
            'prob_refine': prob_refine  # [b,t]
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
    def __init__(self, in_dim=256, n=n_base_sizes, m=n_aspect_ratios):
        super(Head, self).__init__()

        self.in_dim = in_dim
        self.n = n
        self.m = m
        self.num_layers = 3

        self.in_conv = BasicBlock_Conv2D(in_dim=in_dim, out_dim=2*in_dim)

        self.regression_conv = []
        for i in range(self.num_layers):
            self.regression_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.regression_conv = nn.Sequential(*self.regression_conv)

        self.classification_conv = []
        for i in range(self.num_layers):
            self.classification_conv.append(BasicBlock_Conv2D(in_dim, in_dim))
        self.classification_conv = nn.Sequential(*self.classification_conv)

        self.droupout_feat = torch.nn.Dropout(p=0.2)
        self.droupout_cls = torch.nn.Dropout(p=0.2)

        self.regression_head = nn.Conv2d(in_dim, n * m * 4, kernel_size=3, padding=1)
        self.classification_head = nn.Conv2d(in_dim, n * m * 1, kernel_size=3, padding=1)

        self.regression_head.apply(self.init_weights_conv)
        self.classification_head.apply(self.init_weights_conv)



    def init_weights_conv(self, m):
        if type(m) == nn.Conv2d:
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)

    def forward(self, x):
        '''
        x in shape [B, c, h, w]
        '''
        B, c, h, w = x.shape

        feat_reg, feat_cls = self.in_conv(x).split([c, c], dim=1)   # both [B,c,h,w]
        # dpout pos 1, seems better
        feat_reg = self.droupout_feat(feat_reg)
        feat_cls = self.droupout_cls(feat_cls)

        feat_reg = self.regression_conv(feat_reg)        # [B,n*m*4,h,w]
        feat_cls = self.classification_conv(feat_cls)    # [B,n*m*1,h,w]

        # dpout pos 2

        out_reg = self.regression_head(feat_reg)
        out_cls = self.classification_head(feat_cls)

        out_reg = rearrange(out_reg, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=4)
        out_cls = rearrange(out_cls, 'B (n m c) h w -> B (h w n m) c', h=h, w=w, n=self.n, m=self.m, c=1)

        return out_reg, out_cls


class Head_prob(nn.Module):
    def __init__(self, config, h=8, w=8, dim=n_base_sizes*n_aspect_ratios):
        super(Head_prob, self).__init__()

        self.dim = dim
        self.h = h
        self.w = w
        self.config = config
        self.window_transformer = config.model.window_transformer

        # spatial-temporal PE
        pe_3d = torch.zeros(1, config.dataset.clip_num_frames*h*w, self.dim)
        self.pe_3d = nn.parameter.Parameter(pe_3d)

        # feature transformer
        self.temporal_mask = None
        self.feat_corr_transformer = []
        self.num_transformer = 1
        for _ in range(self.num_transformer):
            self.feat_corr_transformer.append(
                    torch.nn.TransformerEncoderLayer(
                        d_model=self.dim, 
                        nhead=1,
                        dim_feedforward=int(4 * self.dim),
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True
                ))
        self.feat_corr_transformer = nn.ModuleList(self.feat_corr_transformer)

        # conv layers
        n_layers = int(math.log2(self.h))
        self.conv_layer = []
        for _ in range(n_layers):
            self.conv_layer.append(nn.Sequential(
                nn.Conv2d(self.dim, self.dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.dim),
            ))
        self.conv_layer = nn.ModuleList(self.conv_layer)

        # final mlp
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.BatchNorm1d(self.dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.dim, 1),
        )
        self.mlp.apply(self.init_weights_linear)


    def init_weights_linear(self, m):
        if type(m) == nn.Linear:
            #nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=1e-6)
            nn.init.normal_(m.bias, mean=0.0, std=1e-6)


    def forward(self, x):
        '''
        x in shape [b,t,N=h*w*n,c=1]
        '''
        b, t = x.shape[:2]

        x = x.squeeze(-1)
        x = rearrange(x, 'b t (h w n) -> b (t h w) n', h=self.h, w=self.w, n=self.dim) #+ self.pe_3d
        
        # mask = self.get_mask(x, t=t)
        # for layer in self.feat_corr_transformer:
        #     x = layer(x, src_mask=mask)
        
        x = rearrange(x, 'b (t h w) n -> (b t) n h w', t=t, h=self.h, w=self.w, n=self.dim)
        for layer in self.conv_layer:
            x = layer(x)
        x = x.squeeze()

        x = self.mlp(x).squeeze()   # [b*t]
        x = rearrange(x, '(b t) -> b t', b=b, t=t)
        return x
    
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
