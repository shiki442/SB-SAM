from model import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append('../')

conv1x1 = layers.ddpm_conv1x1
conv3x3 = layers.ddpm_conv3x3
NIN = layers.NIN
default_init = layers.default_init


class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier embeddings for noise levels."""

    def __init__(self, embedding_size=256, scale=10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size)
                              * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Combine(nn.Module):
    """Combine information from skip connections."""

    def __init__(self, dim1, dim2, method='cat'):
        super().__init__()
        self.Conv_0 = conv1x1(dim1, dim2)
        self.method = method

    def forward(self, x, y):
        h = self.Conv_0(x)
        if self.method == 'cat':
            return torch.cat([h, y], dim=1)
        elif self.method == 'sum':
            return h + y
        else:
            raise ValueError(f'Method {self.method} not recognized.')


class AttnBlockpp(nn.Module):
    """Channel-wise self-attention block. Modified from DDPM."""

    def __init__(self, channels, skip_rescale=False, init_scale=0.):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels,
                                        eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=init_scale)
        self.skip_rescale = skip_rescale

    def forward(self, x):
        B, C, L = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bcl,bci->bli', q, k) * (int(C) ** (-0.5))
        # w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        # w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bli,bci->bcl', w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


class Upsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1), out_padding=0):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch)
        # else:
        #     if with_conv:
        #         self.Conv2d_0 = up_or_downpling.Conv2d(in_ch, out_ch,
        #                                                     kernel=3, up=True,
        #                                                     resample_kernel=fir_kernel,
        #                                                     use_bias=True,
        #                                                     kernel_init=default_init())
        self.fir = fir
        self.with_conv = with_conv
        self.fir_kernel = fir_kernel
        self.out_ch = out_ch
        self.out_padding = out_padding

    def forward(self, x):
        B, C, L = x.shape

        if not self.fir:
            L_out = L * 2 + 1 if self.out_padding == 1 else L * 2
            h = F.interpolate(x, L_out, mode='nearest')
            if self.with_conv:
                h = self.Conv_0(h)
        else:
            pass
            # if not self.with_conv:
            #     h = up_or_downpling.upsample_2d(x, self.fir_kernel, factor=2)
            # else:
            #     h = self.Conv2d_0(x)

        return h


class Downsample(nn.Module):
    def __init__(self, in_ch=None, out_ch=None, with_conv=False, fir=False,
                 fir_kernel=(1, 3, 3, 1)):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        if not fir:
            if with_conv:
                self.Conv_0 = conv3x3(in_ch, out_ch, stride=2, padding=0)
        # else:
        #     if with_conv:
        #         self.Conv2d_0 = up_or_downpling.Conv2d(in_ch, out_ch,
        #                                                 kernel=3, down=True,
        #                                                 resample_kernel=fir_kernel,
        #                                                 use_bias=True,
        #                                                 kernel_init=default_init())
        self.fir = fir
        self.fir_kernel = fir_kernel
        self.with_conv = with_conv
        self.out_ch = out_ch

    def forward(self, x):
        B, C, L = x.shape
        if not self.fir:
            if self.with_conv:
                x = F.pad(x, (0, 1))
                x = self.Conv_0(x)
            else:
                x = F.avg_pool2d(x, 2, stride=2)
        # else:
        #     if not self.with_conv:
        #         x = up_or_downpling.downsample_2d(x, self.fir_kernel, factor=2)
        #     else:
        #         x = self.Conv2d_0(x)

        return x


class ResnetBlockDDPMpp(nn.Module):
    """ResBlock adapted from DDPM."""

    def __init__(self, act, in_ch, out_ch=None, temb_dim=None, conv_shortcut=False,
                 dropout=0.1, skip_rescale=False, init_scale=0.):
        super().__init__()
        out_ch = out_ch if out_ch else in_ch
        self.GroupNorm_0 = nn.GroupNorm(num_groups=min(
            in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.Conv_0 = conv3x3(in_ch, out_ch)
        if temb_dim is not None:
            self.Dense_0 = nn.Linear(temb_dim, out_ch)
            self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
            nn.init.zeros_(self.Dense_0.bias)
        self.GroupNorm_1 = nn.GroupNorm(num_groups=min(
            out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        self.Dropout_0 = nn.Dropout(dropout)
        self.Conv_1 = conv3x3(out_ch, out_ch, init_scale=init_scale)
        if in_ch != out_ch:
            if conv_shortcut:
                self.Conv_2 = conv3x3(in_ch, out_ch)
            else:
                self.NIN_0 = NIN(in_ch, out_ch)

        self.skip_rescale = skip_rescale
        self.act = act
        self.out_ch = out_ch
        self.conv_shortcut = conv_shortcut

    def forward(self, x, temb=None):
        h = self.act(self.GroupNorm_0(x))
        h = self.Conv_0(h)
        if temb is not None:
            h += self.Dense_0(self.act(temb))[:, :, None]
        h = self.act(self.GroupNorm_1(h))
        h = self.Dropout_0(h)
        h = self.Conv_1(h)
        if x.shape[1] != self.out_ch:
            if self.conv_shortcut:
                x = self.Conv_2(x)
            else:
                x = self.NIN_0(x)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)

class Conditioner(nn.Module):
    """Conditioning network for FiLM."""

    def __init__(self, in_dim, embed_dim, act, cond_drop_prob=0.1, batch_norm=False):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.act = act
        self.cond_drop_prob = cond_drop_prob
        self.batch_norm = batch_norm
        self.batch_norm = nn.BatchNorm1d(in_dim, affine=False)

        self.Dense_0 = nn.Linear(in_dim, embed_dim)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
        nn.init.zeros_(self.Dense_0.bias)
        
        self.Dense_1 = nn.Linear(embed_dim, embed_dim)
        self.Dense_1.weight.data = default_init()(self.Dense_1.weight.data.shape)
        nn.init.zeros_(self.Dense_1.bias)

    @staticmethod
    def prob_mask_like(shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

    def forward(self, conditions):
        batch = len(conditions)
        cond_embeds = conditions
        if self.batch_norm:
            cond_embeds = self.batch_norm(cond_embeds)
        if self.cond_drop_prob > 0. and self.training:
            prob_keep_mask = self.prob_mask_like((batch, 1), 1. - self.cond_drop_prob, cond_embeds.device)
            cond_embeds = prob_keep_mask * cond_embeds
        return self.Dense_1(self.act(self.Dense_0(cond_embeds)))

class FiLM(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        act
    ):
        super().__init__()
        self.Dense_0 = nn.Linear(dim, hidden_dim * 4)
        self.Dense_0.weight.data = default_init()(self.Dense_0.weight.data.shape)
        nn.init.zeros_(self.Dense.bias)

        self.Dense_1 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.Dense_1.weight.data = default_init()(self.Dense_1.weight.data.shape)
        nn.init.zeros_(self.Dense_1.bias)

        self.net = nn.Sequential(
            self.Dense_0,
            act,
            self.Dense_1
        )

    def forward(self, conditions, hiddens):
        scale, shift = self.net(conditions).chunk(2, dim = -1)
        return hiddens * (scale + 1) + shift

if __name__ == "__main__":

    act = nn.SiLU()
    toynet = AttnBlockpp(16)
    x = torch.randn(100, 16, 20)
    y = toynet(x)
    print(toynet)
