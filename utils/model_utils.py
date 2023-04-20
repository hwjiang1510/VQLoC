import numpy as np
import torch
import torch.nn as nn
import math
from einops import rearrange


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc
    

def positionalencoding1d(d_model, length):
    """
    positional encoding for 1-d sequence
    :param d_model: dimension of the model (C)
    :param length: length of positions (N)
    :return: length*d_model position matrix, shape [N, C]
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))       # [N,C//2]
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


def positionalencoding2d(d_model, height, width, type='sinusoidal'):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix, shape [H*W, C]
    """
    if type == 'sinusoidal':
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model_origin = d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(1).repeat(1, width, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(1).repeat(1, width, 1)
        pe[d_model::2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, height)
        pe[d_model + 1::2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, height)
        pe = rearrange(pe, 'c h w -> (h w) c')[:,:d_model_origin]
    elif type == 'zero':
        pe = torch.zeros(height * width, d_model)
    return pe


def positionalencoding3d(d_model, height, width, depth, type='sinusoidal'):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :param depth: depth of the positions
    :return: d_model*height*width position matrix, shape [H*W, C]
    """
    if type == 'sinusoidal':
        d_model_interv = int(np.ceil(d_model / 6) * 2)
        if d_model_interv % 2:
            d_model_interv += 1
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model_interv, 2).float() / d_model_interv))
        pos_x = torch.arange(height).type(inv_freq.type())
        pos_y = torch.arange(width).type(inv_freq.type())
        pos_z = torch.arange(depth).type(inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros(height, width, depth, d_model_interv * 3)
        emb[:, :, :, : d_model_interv] = emb_x
        emb[:, :, :, d_model_interv : 2 * d_model_interv] = emb_y
        emb[:, :, :, 2 * d_model_interv :] = emb_z
        emb = rearrange(emb, 'h w d c -> (h w d) c')[:,:d_model]
    elif type == 'zero':
        emb = torch.zeros(height * width * depth, d_model)
    return emb


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


def BasicBlock_Conv2D(in_dim, out_dim):
    module = nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, 3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(inplace=True)
                    )
    return module

def BasicBlock_MLP(dims):
    dims_ = dims[:-1]
    dims1, dims2 = dims_[:-1], dims_[1:]
    mlp = []
    for (dim1, dim2) in zip(dims1, dims2):
        mlp.append(
            nn.Sequential(
                nn.Linear(dim1, dim2),
                nn.BatchNorm1d(dim1),
                nn.LeakyReLU(inplace=True),
        ))
    mlp.append(
        nn.Sequential(
            nn.Linear(dims[-2], dims[-1]),
        ))
    mlp = nn.Sequential(*mlp)
    return mlp
