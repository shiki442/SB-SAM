import math
import torch
from torch import nn
from model.utils import register_model
from model.layers import get_act
from model import layerspp


def L_out(layer, L_in):
    if isinstance(layer, layerspp.Upsample):
        return L_in * 2
    elif isinstance(layer, layerspp.Downsample):
        # L_in = L_in + 1
        # kernel_size = 3
        # stride = 2
        # padding = 0
        # dilation = 1
        return L_in // 2
    elif isinstance(layer, nn.Conv1d):
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
        padding = layer.padding[0]
        dilation = layer.dilation[0]
        return (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    elif isinstance(layer, nn.ConvTranspose1d):
        kernel_size = layer.kernel_size[0]
        stride = layer.stride[0]
        padding = layer.padding[0]
        output_padding = layer.output_padding[0]
        return (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
    else:
        raise NotImplementedError


def _init_params(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.0)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2)
                              * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


@register_model(name='ResNet')
class ResNet(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        assert cfg.net.depth >= 1, 'hidden_depth must be greater than 0!'
        embed_dim = cfg.net.width
        self.hidden_depth = cfg.net.depth
        self.use_bn = cfg.net.use_bn
        # time embedding
        self.embed = nn.Sequential(GaussianFourierProjection(
            embed_dim=embed_dim), nn.Linear(embed_dim, embed_dim))
        # fc layers
        self.input = nn.Linear(cfg.data.nf, embed_dim)
        self.fc_all = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for i in range(self.hidden_depth)])
        self.output = nn.Linear(embed_dim, cfg.data.nf)
        # batch normalization
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(num_features=embed_dim) for i in range(self.hidden_depth)])
        # The swish activation function
        self.act = get_act(cfg)
        self.sigma_min = cfg.model.sigma_min
        self.sigma_max = cfg.model.sigma_max
        self.apply(_init_params)

    def forward(self, x, t):
        original_shape = x.shape
        x_f = torch.flatten(x, start_dim=-2)
        h = self.input(x_f) + self.embed(t)
        # residue connections
        for i in range(self.hidden_depth):
            h = h + self.act(self.fc_all[i](h))
            if self.use_bn:
                h = self.bn[i](h)
        h = self.output(h)

        return h.view(original_shape)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]


@register_model(name='Unet')
class Unet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        channels = [x * cfg.model.nf for x in cfg.model.ch_mult]
        L_outs = []
        # time embedding
        self.embed = nn.Sequential(GaussianFourierProjection(
            embed_dim=cfg.net.width), nn.Linear(cfg.net.width, cfg.net.width))
        # fc layers
        self.conv1 = nn.Conv1d(
            cfg.data.d, channels[0], 3, stride=1, bias=False)
        L_outs.append(L_out(self.conv1, cfg.data.n))
        self.dense1 = Dense(cfg.net.width, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv1d(
            channels[0], channels[1], 3, stride=2, bias=False)
        L_outs.append(L_out(self.conv2, L_outs[-1]))
        self.dense2 = Dense(cfg.net.width, channels[1])
        self.gnorm2 = nn.GroupNorm(16, num_channels=channels[1])
        self.conv3 = nn.Conv1d(
            channels[1], channels[2], 3, stride=2, bias=False)
        L_outs.append(L_out(self.conv3, L_outs[-1]))
        self.dense3 = Dense(cfg.net.width, channels[2])
        self.gnorm3 = nn.GroupNorm(16, num_channels=channels[2])
        self.conv4 = nn.Conv1d(
            channels[2], channels[3], 3, stride=2, bias=False)
        L_outs.append(L_out(self.conv4, L_outs[-1]))
        self.dense4 = Dense(cfg.net.width, channels[3])
        self.gnorm4 = nn.GroupNorm(16, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose1d(
            channels[3], channels[2], 3, stride=2, bias=False)
        if L_out(self.tconv4, L_outs.pop()) != L_outs[-1]:
            self.tconv4 = nn.ConvTranspose1d(
                channels[3], channels[2], 3, stride=2, bias=False, output_padding=1)
        self.dense5 = Dense(cfg.net.width, channels[2])
        self.tgnorm4 = nn.GroupNorm(16, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose1d(
            channels[2] + channels[2], channels[1], 3, stride=2, bias=False)
        if L_out(self.tconv3, L_outs.pop()) != L_outs[-1]:
            self.tconv3 = nn.ConvTranspose1d(
                channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(cfg.net.width, channels[1])
        self.tgnorm3 = nn.GroupNorm(16, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose1d(
            channels[1] + channels[1], channels[0], 3, stride=2, bias=False)
        if L_out(self.tconv2, L_outs.pop()) != L_outs[-1]:
            self.tconv2 = nn.ConvTranspose1d(
                channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(cfg.net.width, channels[0])
        self.tgnorm2 = nn.GroupNorm(16, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose1d(
            channels[0] + channels[0], cfg.data.d, 3, stride=1)

        # The activation function
        self.act = get_act(cfg)
        # self.marginal_prob_std = marginal_prob_std
        # self.apply(_init_params)

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        # Incorporate information from t
        h1 += self.dense1(embed)
        # Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        # Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        # h = h / self.marginal_prob_std(t)[:, None, None]
        return h


class TimeIndependentScoreNet(nn.Module):

    def __init__(self, x_dim, hidden_depth=2, embed_dim=128, use_bn=True, cfg=None):
        assert hidden_depth >= 1, 'hidden_depth must be greater than 0!'
        super().__init__()
        self.hidden_depth = hidden_depth
        self.use_bn = use_bn
        # fc layers
        self.input = nn.Linear(x_dim, embed_dim)
        self.fc_all = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for i in range(self.hidden_depth)])
        self.output = nn.Linear(embed_dim, x_dim)
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(num_features=embed_dim) for i in range(self.hidden_depth)])
        # The swish activation function
        self.act = nn.SiLU()
        self.cfg = cfg
        self.apply(_init_params)

    def forward(self, x):
        h = self.input(x)
        # residue connections
        for i in range(self.hidden_depth):
            h = h + self.act(self.fc_all[i](h))
            if self.use_bn:
                h = self.bn[i](h)
        h = self.output(h)
        return h

    def div(self, x):
        divergence = torch.zeros(x.shape[0], device=self.cfg.device)
        score = self(x)
        for i in range(score.shape[1]):
            score_i = score[:, i]
            grad_i = torch.autograd.grad(
                score_i, x, grad_outputs=torch.ones_like(score_i), retain_graph=True)[0]
            divergence += grad_i[:, i]
        return divergence
