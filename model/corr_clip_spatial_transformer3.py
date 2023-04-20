import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import Block
#from model.local_transformer import Block_local
from utils.model_utils import PositionalEncoding1D, positionalencoding1d, positionalencoding3d, positionalencoding2d
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

        self.window_transformer = config.model.window_transformer
        self.resolution_transformer = config.model.resolution_transformer
        self.topk = 64 #self.resolution_transformer ** 2

        # feature reduce layer
        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )
        
        # clip-query correspondence
        self.CQ_corr_transformer = make_transformer_decoder_layers(n_layers=1, d_model=256, nhead=8, dim_feedforward=1024, 
                                                                   dropout=0.0, activation='gelu', batch_first=True)

        # self-refinement
        self.feat_refine_transformer = make_transformer_encoder_layers(n_layers=1, d_model=256, nhead=8, dim_feedforward=1024, 
                                                                   dropout=0.0, activation='gelu', batch_first=True)
        
        # spatial-temporal local-window fusion
        self.feat_fusion_transformer = make_transformer_encoder_layers(n_layers=3, d_model=256, nhead=8, dim_feedforward=1024, 
                                                                   dropout=0.0, activation='gelu', batch_first=True)
        
        # positional embedding
        self.pe_2d = torch.zeros(1, self.clip_feat_size_coarse**2, 256)    # [1,h*w,c]
        self.pe_3d = torch.zeros(1, config.dataset.clip_num_frames, self.clip_feat_size_coarse**2, 256)   # [1,t,h*w,c]
        self.pe_2d = nn.parameter.Parameter(self.pe_2d)
        self.pe_3d = nn.parameter.Parameter(self.pe_3d)

        # spatial-temporal transformer mask
        self.temporal_mask = None

        # output prediction token
        self.out_token = nn.parameter.Parameter(0.02 * torch.randn(1,1,256))
        self.out_transformer = make_transformer_decoder_layers(n_layers=1, d_model=256, nhead=8, dim_feedforward=1024, 
                                                                   dropout=0.0, activation='gelu', batch_first=True)

        # output head
        self.out = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 5),
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

        # decrease feature channel size
        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)     # [b,c,h,w] / [b*t,c,h,w]
        
        # clip-query fusion transformer
        query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')  # [b*t,n=h*w,c]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')                                         # [b*t,n,c]
        for layer in self.CQ_corr_transformer:
            clip_feat = layer(clip_feat, query_feat)                                                     # [b*t,n,c]

        # get topk feature and index
        c = clip_feat.shape[-1]
        affinity_matrix = clip_feat @ query_feat.permute(0,2,1)                                 # [b*t,n,n]
        affinity_score = affinity_matrix.sum(dim=-1)                                            # [b*t,n]
        _, idx_topk = torch.topk(affinity_score, k=self.topk)                                   # [b*t,k]
        feat_topk = torch.gather(clip_feat, dim=1, index=idx_topk.unsqueeze(-1).repeat(1,1,c))  # [b*t,k,c]

        # self-attention tansformer
        pe2d = torch.gather(self.pe_2d.repeat(b*t,1,1), dim=1, index=idx_topk.unsqueeze(-1).repeat(1,1,c))  # [b*t,k,c]
        feat_topk += pe2d
        for layer in self.feat_refine_transformer:
            feat_topk = layer(feat_topk)    # [b*t,k,c]
        feat_topk = rearrange(feat_topk, '(b t) k c -> b (t k) c', b=b, t=t)

        # temporal local window self-attention transformer
        pe3d = self.get_pe3d(self.pe_3d, idx_topk, b)   # [b,t*k,c]
        feat_topk += pe3d                               # [b,t*k,c]
        mask = self.get_mask(feat_topk, t)              # [t*k,t*k]
        for layer in self.feat_fusion_transformer:
            feat_topk = layer(feat_topk, src_mask=mask)

        # get prediction
        feat_topk = rearrange(feat_topk, 'b (t k) c -> (b t) k c', t=t)
        out_token = self.out_token.clone().repeat(b*t, 1, 1)    # [b*t,1,c]
        for layer in self.out_transformer:
            out_token = layer(out_token, feat_topk)     # [b*t,1,256]
        out_token = out_token.squeeze()
        pred = self.out(out_token)      # [b*t,c']
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
    

    def get_mask(self, src, t):
        '''
        src: feature in shape [b, t*k, c]
        return: local window mask [tk, tk]
        '''
        if not torch.is_tensor(self.temporal_mask):
            k = src.shape[1] // t
            tk = src.shape[1]
            mask = torch.ones(tk, tk).float() * float('-inf')
            window_size = self.window_transformer // 2
            for i in range(t):
                min_idx = max(0, (i-window_size)*k)
                max_idx = min(tk, (i+window_size+1)*k)
                mask[i*k: (i+1)*k, min_idx: max_idx] = 0.0
            mask = mask.to(src.device)
            self.temporal_mask = mask
        return self.temporal_mask
    

    def get_pe3d(self, pe3d, idx, b):
        '''
        pe3d: raw pe3d in shape [1,t,n=h*w,c]
        idx: in shape [b*t,k]
        return: pe in shape [b,t*k,c]
        '''
        c = pe3d.shape[-1]
        pe3d = pe3d.repeat(b,1,1,1)   # [b,t,h*w,c]
        pe3d = rearrange(pe3d, 'b t n c -> (b t) n c')    # [b*t,n,c]
        pe3d = torch.gather(pe3d, dim=1, index=idx.unsqueeze(-1).repeat(1,1,c))     # [b*t,k,c]
        pe3d = rearrange(pe3d, '(b t) k c -> b (t k) c', b=b)
        return pe3d



def make_transformer_encoder_layers(n_layers, d_model=256, nhead=8, dim_feedforward=1024, 
                                    dropout=0.0, activation='gelu', batch_first=True):
    module = []
    for _ in range(n_layers):
        module.append(
            torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first
            )
        )
    return nn.ModuleList(module)


def make_transformer_decoder_layers(n_layers, d_model=256, nhead=8, dim_feedforward=1024, 
                                    dropout=0.0, activation='gelu', batch_first=True):
    module = []
    for _ in range(n_layers):
        module.append(
            torch.nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=batch_first
            )
        )
    return nn.ModuleList(module)