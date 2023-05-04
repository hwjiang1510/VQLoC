import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import Block
from utils.model_utils import PositionalEncoding1D, positionalencoding1d, positionalencoding3d, positionalencoding2d
from utils.model_utils import BasicBlock_Conv2D, BasicBlock_MLP
from utils.anchor_utils import generate_anchor_boxes_on_regions
from einops import rearrange
import math
from model.corr_clip_spatial_transformer2_anchor_2heads import build_backbone
from model.transformer import Attention


class Minotaur(nn.Module):
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

        # feature reduce layer
        self.reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.transformer_encoder = []
        for _ in range(6):
            self.transformer_encoder.append(
                torch.nn.TransformerEncoderLayer(
                        d_model=256, 
                        nhead=8,
                        dim_feedforward=2048,
                        dropout=0.0,
                        activation='gelu',
                        batch_first=True
                ))
        self.transformer_encoder = nn.ModuleList(self.transformer_encoder)
        
        self.embedding = torch.zeros(config.dataset.clip_num_frames, 256).unsqueeze(0)  # [1,T,256]
        self.embedding = nn.parameter.Parameter(self.embedding)

        self.transformer_decoder = []
        for i in range(6):
            self.transformer_decoder.append(TubeDetrLayer(config, dim=256))
        self.transformer_decoder = nn.ModuleList(self.transformer_decoder)

        self.MLP = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 5)
        )
        self.MLP.apply(self.init_weights_linear)


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

        all_feat = torch.cat([query_feat, clip_feat], dim=0)
        all_feat = self.reduce(all_feat)
        query_feat, clip_feat = all_feat.split([b, b*t], dim=0)
        h, w = clip_feat.shape[-2:]
        
        query_feat = rearrange(query_feat.unsqueeze(1).repeat(1,t,1,1,1), 'b t c h w -> (b t) (h w) c')  # [b*t,n,c]
        clip_feat = rearrange(clip_feat, 'b c h w -> b (h w) c')                                         # [b*t,n,c]
        clip_feat = torch.cat([clip_feat, query_feat], dim=1)                                            # [b*t,n',c]

        for layer in self.transformer_encoder:
            clip_feat = layer(clip_feat)

        clip_feat = rearrange(clip_feat, '(b t) n c -> b (t n) c', b=b, t=t, n=2*h*w)
                            
        embedding = self.embedding.repeat(b,1,1)    # [b,t,c]
        for layer in self.transformer_decoder:
            embedding = layer(embedding, clip_feat)
        
        embedding = rearrange(embedding, 'b t c -> (b t) c')
        pred = self.MLP(embedding)         # [b*t,5]
        pred = rearrange(pred, '(b t) c -> b t c', b=b, t=t)
        center, hw, prob = pred.split([2, 2, 1], dim=-1)
        bbox = torch.cat([center - hw, center + hw], dim=-1)
        
        result = {
            'center': center,
            'hw': hw,
            'bbox': bbox,
            'prob': prob.squeeze(-1)    # [b,t]
        }
        return result



class TubeDetrLayer(nn.Module):
    def __init__(self, config, dim=256):
        super().__init__()

        self.config = config
        self.dim = dim

        self.attention = Attention(dim=dim, num_heads=4)

        self.mask = None
        self.transformer_deocoder_layer = torch.nn.TransformerDecoderLayer(
                                                        d_model=self.dim,
                                                        nhead=4,
                                                        dim_feedforward=1024,
                                                        dropout=0.0,
                                                        activation='gelu',
                                                        batch_first=True)
        
    def forward(self, embedding, feat):
        '''
        embedding in [B,T,C]
        feat in [B,T*N,C]
        '''
        embedding = embedding + self.attention(embedding, embedding, embedding)

        mask = self.get_mask(embedding, feat)
        embedding = self.transformer_deocoder_layer(embedding, feat, memory_mask=mask)
        return embedding
    

    def get_mask(self, embedding, feat):
        if not torch.is_tensor(self.mask):
            t = embedding.shape[1]
            tn = feat.shape[1]
            n = tn // t
            mask = torch.ones(t, tn).float() * float('-inf')

            for i in range(t):
                min_idx = max(0, i*n)
                max_idx = min(tn, (i+1)*n)
                mask[i, min_idx: max_idx] = 0.0
            mask = mask.to(embedding.device)
            self.mask = mask
        return self.mask

        



