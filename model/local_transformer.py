import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from model.transformer import Mlp

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from localAttention import (similar_forward,
                            similar_backward,
                            weighting_forward,
                            weighting_backward_ori,
                            weighting_backward_weight)

__all__ = ['f_similar', 'f_weighting', 'LocalAttention', 'TorchLocalAttention']


class similarFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_loc, kH, kW):
        ctx.save_for_backward(x_ori, x_loc)
        ctx.kHW = (kH, kW)
        output = similar_forward(x_ori, x_loc, kH, kW)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_loc = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = similar_backward(x_loc, grad_outputs, kH, kW, True)
        grad_loc = similar_backward(x_ori, grad_outputs, kH, kW, False)

        return grad_ori, grad_loc, None, None


class weightingFunction(Function):
    @staticmethod
    def forward(ctx, x_ori, x_weight, kH, kW):
        ctx.save_for_backward(x_ori, x_weight)
        ctx.kHW = (kH, kW)
        output = weighting_forward(x_ori, x_weight, kH, kW)

        return output

    @staticmethod
    #@once_differentiable
    def backward(ctx, grad_outputs):
        x_ori, x_weight = ctx.saved_tensors
        kH, kW = ctx.kHW
        grad_ori = weighting_backward_ori(x_weight, grad_outputs, kH, kW)
        grad_weight = weighting_backward_weight(x_ori, grad_outputs, kH, kW)

        return grad_ori, grad_weight, None, None


f_similar = similarFunction.apply
f_weighting = weightingFunction.apply


class LocalAttention(nn.Module):
    def __init__(self, dim, kH, kW=1):
        super(LocalAttention, self).__init__()
        '''
        num_head = 1
        '''
        self.scale = dim ** -0.5
        self.kH = kH
        self.kW = kW

    def forward(self, query, key, value):
        '''
        query, key and value are in shape [B,C,H,W]
        '''
        
        weight = f_similar(query, key, self.kH, self.kW)
        weight = F.softmax(weight, -1) * self.scale
        out = f_weighting(value, weight, self.kH, self.kW)
        return out


class Block_local(nn.Module):
    def __init__(self, dim, kH, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, return_attn=False):
        super().__init__()

        self.channels = dim

        self.encode_query = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.attn = LocalAttention(dim, kH=kH, kW=1)
        
        self.encode_value = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        self.norm = norm_layer(dim)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor)
    
    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, n = query.shape

        q = self.with_pos_embed(query, query_embed)
        k = self.with_pos_embed(key, key_embed)

        q = self.norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.norm(k.permute(0, 2, 1)).permute(0, 2, 1)

        v = self.encode_value(key).view(b, self.channels, -1).unsqueeze(-1)   # [b,c,n,1]
        q = self.encode_query(q).view(b, self.channels, -1).unsqueeze(-1)
        k = self.encode_key(k).view(b, self.channels, -1).unsqueeze(-1)
        assert len(v.shape) == 4 and len(q.shape) == 4 and len(k.shape) == 4
        assert v.shape[-1] == 1 and q.shape[-1] == 1 and k.shape[-1] == 1

        query = query.view(b, self.channels, -1)
        query = query + self.attn(query=q, key=k, value=v).squeeze()        # [b,c,n]
        query = query.permute(0, 2, 1)

        query = query + self.mlp(self.norm2(query))                         # [b,n,c]
        query = query.permute(0, 2, 1).contiguous().view(b, self.channels, -1)

        return query
