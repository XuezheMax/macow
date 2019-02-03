__author__ = 'max'

from overrides import overrides
from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn.modules.utils import _pair

from macow.flows.flow import Flow
from macow.nnet.weight_norm import Conv2dWeightNorm, NIN2d, NIN4d
from macow.nnet.attention import MultiHeadAttention2d


class NICEConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, s_channels, dilation):
        super(NICEConvBlock, self).__init__()
        self.conv1 = Conv2dWeightNorm(in_channels + s_channels, hidden_channels, kernel_size=3, dilation=dilation, padding=dilation, bias=True)
        self.conv2 = Conv2dWeightNorm(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        self.conv3 = Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation, bias=True)
        self.activation = nn.ELU(inplace=True)

    def init(self, x, s=None, init_scale=1.0):
        if s is not None:
            x = torch.cat([x, s], dim=1)

        out = self.activation(self.conv1.init(x, init_scale=init_scale))

        out = self.activation(self.conv2.init(out, init_scale=init_scale))

        out = self.conv3.init(out, init_scale=0.0)

        return out

    def forward(self, x, s=None):
        if s is not None:
            x = torch.cat([x, s], dim=1)

        out = self.activation(self.conv1(x))

        out = self.activation(self.conv2(out))

        out = self.conv3(out)
        return out


class SelfAttnLayer(nn.Module):
    def __init__(self, channels, heads):
        super(SelfAttnLayer, self).__init__()
        self.attn = MultiHeadAttention2d(channels, heads)
        self.gn = nn.GroupNorm(heads, channels)

    def forward(self, x, pos_enc=None):
        return self.gn(self.attn(x, pos_enc=pos_enc))

    def init(self, x, pos_enc=None, init_scale=1.0):
        return self.gn(self.attn.init(x, pos_enc=pos_enc, init_scale=init_scale))


class NICESelfAttnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, s_channels, slice, heads, pos_enc=True):
        super(NICESelfAttnBlock, self).__init__()
        self.nin1 = NIN2d(in_channels + s_channels, hidden_channels, bias=True)
        self.attn = SelfAttnLayer(hidden_channels, heads)
        self.nin2 = NIN4d(hidden_channels, hidden_channels, bias=True)
        self.activation = nn.ELU(inplace=True)
        self.nin3 = NIN2d(hidden_channels, out_channels, bias=True)
        self.slice_height, self.slice_width = slice
        # positional enc
        if pos_enc:
            self.register_buffer('pos_enc', torch.zeros(hidden_channels, self.slice_height, self.slice_width))
            pos_enc = np.array([[[(h * self.slice_width + w) / np.power(10000, 2.0 * (i // 2) / hidden_channels)
                                  for i in range(hidden_channels)]
                                 for w in range(self.slice_width)]
                                for h in range(self.slice_height)])
            pos_enc[:, :, 0::2] = np.sin(pos_enc[:, :, 0::2])
            pos_enc[:, :, 1::2] = np.cos(pos_enc[:, :, 1::2])
            pos_enc = np.transpose(pos_enc, (2, 0, 1))
            self.pos_enc.copy_(torch.from_numpy(pos_enc).float())
        else:
            self.register_buffer('pos_enc', None)

    def forward(self, x, s=None):
        # [batch, in+s, height, width]
        bs, _, height, width = x.size()
        if s is not None:
            x = torch.cat([x, s], dim=1)

        # slice2d
        # [batch*fh*fw, hidden, slice_heigth, slice_width]
        x = self.slice2d(x, self.slice_height, self.slice_width, init=False)
        x = self.attn(x, pos_enc=self.pos_enc)

        # unslice2d
        # [batch, hidden, height, width]
        x = self.unslice2d(x, height, width, init=False)
        # compute output
        # [batch, out, height, width]
        out = self.nin3(x)
        return out

    def init(self, x, s=None, init_scale=1.0):
        # [batch, in+s, height, width]
        bs, _, height, width = x.size()
        if s is not None:
            x = torch.cat([x, s], dim=1)

        # slice2d
        # [batch*fh*fw, hidden, slice_heigth, slice_width]
        x = self.slice2d(x, self.slice_height, self.slice_width, init=True, init_scale=init_scale)
        x = self.attn.init(x, pos_enc=self.pos_enc, init_scale=init_scale)

        # unslice2d
        # [batch, hidden, height, width]
        x = self.unslice2d(x, height, width, init=True, init_scale=init_scale)
        # compute output
        # [batch, out, height, width]
        out = self.nin3.init(x, init_scale=0.0)
        return out

    def slice2d(self, x: torch.Tensor, slice_height, slice_width, init, init_scale=1.0) -> torch.Tensor:
        batch, n_channels, height, width = x.size()
        assert height % slice_height == 0 and width % slice_width == 0
        fh = height // slice_height
        fw = width // slice_width

        # [batch, channels, height, width] -> [batch, channels, factor_height, slice_height, factor_width, slice_width]
        x = x.view(-1, n_channels, fh, slice_height, fw, slice_width)
        # [batch, channels, factor_height, slice_height, factor_width, slice_width] -> [batch, factor_height, factor_width, channels, slice_height, slice_width]
        x = x.permute(0, 2, 4, 1, 3, 5)
        # [batch, factor_height, factor_width, hidden, slice_height, slice_width]
        x = self.nin1.init(x, init_scale=init_scale) if init else self.nin1(x)
        # [batch * factor_height * factor_width, hidden, slice_height, slice_width]
        hidden_channels = x.size(3)
        x = x.view(-1, hidden_channels, slice_height, slice_width)
        return x

    def unslice2d(self, x: torch.Tensor, height, width, init, init_scale=1.0) -> torch.Tensor:
        _, n_channels, slice_height, slice_width = x.size()
        assert height % slice_height == 0 and width % slice_width == 0
        fh = height // slice_height
        fw = width // slice_width

        # [batch, factor_height, factor_width, channels, slice_height, slice_width]
        x = x.view(-1, fh, fw, n_channels, slice_height, slice_width)
        # [batch, factor_height, factor_width, channels, slice_height, slice_width] -> [batch, channels, factor_height, slice_height, factor_width, slice_width]
        x = x.permute(0, 3, 1, 4, 2, 5)
        # [batch, channels, factor_height, slice_height, factor_width, slice_width]
        x = self.nin2.init(x, init_scale=init_scale) if init else self.nin2(x)
        x = self.activation(x)
        # [batch, channels, height, width]
        x = x.view(-1, n_channels, height, width)
        return x


class NICE(Flow):
    def __init__(self, in_channels, hidden_channels=None, s_channels=None, scale=True, inverse=False, factor=2,
                 type='conv', slice=None, heads=1, pos_enc=True):
        super(NICE, self).__init__(inverse)
        self.in_channels = in_channels
        self.scale = scale
        if hidden_channels is None:
            hidden_channels = min(8 * in_channels, 512)
        out_channels = in_channels // factor
        in_channels = in_channels - out_channels
        self.z1_channels = in_channels
        if scale:
            out_channels = out_channels * 2
        if s_channels is None:
            s_channels = 0
        assert type in ['conv', 'self_attn']
        if type == 'conv':
            self.net = NICEConvBlock(in_channels, out_channels, hidden_channels, s_channels, dilation=1)
        else:
            assert slice is not None, 'slice should be given.'
            slice = _pair(slice)
            self.net = NICESelfAttnBlock(in_channels, out_channels, hidden_channels, s_channels, slice=slice, heads=heads, pos_enc=pos_enc)

    def calc_mu_and_scale(self, z1: torch.Tensor, s=None):
        mu = self.net(z1, s=s)
        scale = None
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            scale = log_scale.add_(2.).sigmoid_()
        return mu, scale

    def init_net(self, z1: torch.Tensor, s=None, init_scale=1.0):
        mu = self.net.init(z1, s=s, init_scale=init_scale)
        scale = None
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            scale = log_scale.add_(2.).sigmoid_()
        return mu, scale

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            s: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        # [batch, in_channels, H, W]
        z1 = input[:, :self.z1_channels]
        z2 = input[:, self.z1_channels:]
        mu, scale = self.calc_mu_and_scale(z1, s)
        if self.scale:
            z2 = z2.mul(scale)
            logdet = scale.log().view(z1.size(0), -1).sum(dim=1)
        else:
            logdet = z1.new_zeros(z1.size(0))
        z2 = z2 + mu
        return torch.cat([z1, z2], dim=1), logdet

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            s: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        z1 = input[:, :self.z1_channels]
        z2 = input[:, self.z1_channels:]
        mu, scale = self.calc_mu_and_scale(z1, s)
        z2 = z2 - mu
        if self.scale:
            z2 = z2.div(scale + 1e-12)
            logdet = scale.log().view(z1.size(0), -1).sum(dim=1) * -1.0
        else:
            logdet = z1.new_zeros(z1.size(0))

        return torch.cat([z1, z2], dim=1), logdet

    @overrides
    def init(self, data: torch.Tensor, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, in_channels, H, W]
        z1 = data[:, :self.z1_channels]
        z2 = data[:, self.z1_channels:]
        mu, scale = self.init_net(z1, s=s, init_scale=init_scale)
        if self.scale:
            z2 = z2.mul(scale)
            logdet = scale.log().view(z1.size(0), -1).sum(dim=1)
        else:
            logdet = z1.new_zeros(z1.size(0))
        z2 = z2 + mu

        return torch.cat([z1, z2], dim=1), logdet

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}, scale={}'.format(self.inverse, self.in_channels, self.scale)

    @classmethod
    def from_params(cls, params: Dict) -> "NICE":
        return NICE(**params)


NICE.register('nice')
