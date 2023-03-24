import torch.nn as nn
import torch
import torch.nn.functional as F
from model.transformer import Block
#from model.local_transformer import Block_local
from utils.model_utils import PositionalEncoding1D, positionalencoding1d
from einops import rearrange
import math
import time


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
        if self.type_transformer == 'local':
            self.window_transformer = config.model.window_transformer

        self.clip_reduce = nn.Sequential(
            nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.query_target_size = []
        self.down_query_heads = []
        for i in range(int(math.log2(self.query_feat_size))):
            self.query_target_size.append((2**i, 2**i))                 # [(1,1), (2,2), (4,4), ..., (query_feat_size//2, query_feat_size//2)]
            self.down_query_heads.append(nn.Sequential(
                    nn.Conv2d(self.backbone_dim, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(inplace=True),
                ))
        self.down_query_heads = nn.ModuleList(self.down_query_heads)

        self.num_head_layers = int(math.log2(self.clip_feat_size_coarse))
        self.down_heads = []
        for _ in range(self.num_head_layers):
            self.down_heads.append(
                nn.Sequential(
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(inplace=True),
            ))
        self.down_heads = nn.ModuleList(self.down_heads)

        self.pe_1d = positionalencoding1d(d_model=512, length=config.dataset.clip_num_frames).unsqueeze(0).permute(0,2,1) #PositionalEncoding1D(channels=512)
        
        self.transformer_layer = []
        self.num_transformer = config.model.num_transformer
        for _ in range(self.num_transformer):
            if self.type_transformer == 'global':
                self.transformer_layer.append(Block(dim=512, num_heads=4, mlp_ratio=4.))
            # elif self.type_transformer == 'local':
            #     self.transformer_layer.append(Block_local(dim=512, kH=self.window_transformer, mlp_ratio=4.))

        self.transformer_layer = nn.ModuleList(self.transformer_layer)

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
        end = time.time()
        b, t, _, h, w = clip.shape
        h2, w2 = query.shape[-2:]
        clip = rearrange(clip, 'b t c h w -> (b t) c h w')

        if fix_backbone:
            with torch.no_grad():
                query_feat = self.extract_feature(query)
                clip_feat = self.extract_feature(clip)
        else:
            query_feat = self.extract_feature(query)
            clip_feat = self.extract_feature(clip)

        clip_feat = self.clip_reduce(clip_feat)
        if [clip_feat.shape[-2], clip_feat.shape[-1]] != [self.clip_feat_size_coarse, self.clip_feat_size_coarse]:
            clip_feat = F.interpolate(clip_feat, (self.clip_feat_size_coarse, self.clip_feat_size_coarse), mode='bilinear')

        query_down = torch.zeros(b, 256, self.clip_feat_size_coarse, self.clip_feat_size_coarse).to(clip_feat.device)
        for size, down_head in zip(self.query_target_size, self.down_query_heads):
            cur_query = down_head(F.interpolate(query_feat, size, mode='bilinear'))
            repeat = (1, 1, self.clip_feat_size_coarse // size[0], self.clip_feat_size_coarse // size[1])
            cur_query = cur_query.repeat(repeat)
            query_down += cur_query
        query_down = query_down.unsqueeze(1).repeat(1,t,1,1,1)
        query_feat = rearrange(query_down, 'b t c h w -> (b t) c h w')

        feat = torch.cat([clip_feat, query_feat], dim=1)        
        for cur_head in self.down_heads:
            cur_shape = feat.shape[2:]
            feat = cur_head(F.interpolate(feat, (cur_shape[0]//2, cur_shape[1]//2), mode='bilinear'))

        token = feat.squeeze()      # [b*t,512]
        # #token = rearrange(token, '(b t) c -> b t c', b=b, t=t)
        # # token = self.pe_1d(token)
        # # token = rearrange(token, 'b t c -> b c t')
        # token = rearrange(token, '(b t) c -> b c t', b=b, t=t)
        # for layer in self.transformer_layer:
        #     token = layer(query=token, key=token, query_embed=self.pe_1d, key_embed=self.pe_1d)
        # token = rearrange(token, 'b c t -> (b t) c')

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