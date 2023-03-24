
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class Block(nn.Module):
    def __init__(self, dim, num_heads=1, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm, return_attn=False):
        super().__init__()

        self.channels = dim

        self.encode_query = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_key = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.encode_value = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)
        self.norm = norm_layer(dim)

        self.attn = Attention(dim, num_heads=num_heads)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor)
    
    def get_attn(self, query, key, query_embed=None, key_embed=None):
        b, c, n = query.shape

        q = self.with_pos_embed(query, query_embed)
        k = self.with_pos_embed(key, key_embed)

        q = self.norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.norm(k.permute(0, 2, 1)).permute(0, 2, 1)

        q = self.encode_query(q).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)
        k = self.encode_key(k).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)
        return self.attn.get_attn(query=q, key=k)   # [b,n,n]
    
    def forward(self, query, key, query_embed=None, key_embed=None):
        b, c, n = query.shape

        q = self.with_pos_embed(query, query_embed)
        k = self.with_pos_embed(key, key_embed)

        q = self.norm(q.permute(0, 2, 1)).permute(0, 2, 1)
        k = self.norm(k.permute(0, 2, 1)).permute(0, 2, 1)

        v = self.encode_value(key).view(b, self.channels, -1)
        v = v.permute(0, 2, 1)

        q = self.encode_query(q).view(b, self.channels, -1)
        q = q.permute(0, 2, 1)

        k = self.encode_key(k).view(b, self.channels, -1)
        k = k.permute(0, 2, 1)

        query = query.view(b, self.channels, -1).permute(0, 2, 1)
        query = query + self.attn(query=q, key=k, value=v)

        query = query + self.mlp(self.norm2(query))
        query = query.permute(0, 2, 1).contiguous().view(b, self.channels, -1)

        return query


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

    def get_attn(self, query, key):
        B, N, C = query.shape
        attn = torch.matmul(query, key.transpose(-2, -1)) #* self.scale
        attn = attn.softmax(dim=-1)
        #__import__('pdb').set_trace()
        return attn
    
    def forward(self, query, key, value):
        B, N, C = query.shape
        query = query.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        key = key.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        value = value.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.matmul(attn, value).transpose(1, 2).reshape(B, N, C)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        # nn.init.xavier_uniform_(self.fc1.weight)
        # nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.normal_(self.fc1.bias, std=1e-4)
        nn.init.normal_(self.fc2.bias, std=1e-4)
