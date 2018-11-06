__author__ = 'max'

from overrides import overrides
from typing import Tuple, Dict
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.nnet import Conv2dWeightNorm


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, dropout=0.0):
        super(ResNetBlock, self).__init__()
        self.conv1 = Conv2dWeightNorm(in_channels, hidden_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = Conv2dWeightNorm(hidden_channels, hidden_channels, kernel_size=1, bias=True)
        self.conv3 = Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.downsample = Conv2dWeightNorm(in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def init(self, x, init_scale=1.0):
        residual = self.downsample.init(x, init_scale=0.0)

        out = self.conv1.init(x, init_scale=init_scale)
        out = self.activation(out)

        out = self.conv2.init(out, init_scale=init_scale)
        out = self.activation(out)

        # dropout
        out = self.dropout(out)

        out = self.conv3.init(out, init_scale=0.0)

        return out + residual

    def forward(self, x):
        residual = self.downsample(x)

        out = self.activation(self.conv1(x))

        out = self.activation(self.conv2(out))

        # dropout
        out = self.dropout(out)

        out = self.conv3(out) + residual
        return out


class NICE(Flow):
    def __init__(self, in_channels, hidden_channels=None, scale=True, inverse=False, dropout=0.0):
        super(NICE, self).__init__(inverse)
        self.in_channels = in_channels
        self.scale = scale
        if hidden_channels is None:
            hidden_channels = 8 * in_channels
        out_channels = in_channels // 2
        in_channels = in_channels - out_channels
        if scale:
            out_channels = out_channels * 2
        self.net = ResNetBlock(in_channels, out_channels, hidden_channels=hidden_channels, dropout=dropout)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        # [batch, in_channels, H, W]
        z1, z2 = input.chunk(2, dim=1)
        mu = self.net(z1)
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            z2 = z2.mul(log_scale.exp())
            logdet = log_scale.view(z1.size(0), -1).sum(dim=1)
        else:
            logdet = z1.new_zeros(z1.size(0))
        z2 = z2 + mu

        return torch.cat([z1, z2], dim=1), logdet

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        z1, z2 = input.chunk(2, dim=1)
        mu = self.net(z1)
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            z2 = (z2 - mu).div(log_scale.exp() + 1e-12)
            logdet = log_scale.view(z1.size(0), -1).sum(dim=1) * -1.0
        else:
            z2 = z2 - mu
            logdet = z1.new_zeros(z1.size(0))

        return torch.cat([z1, z2], dim=1), logdet

    @overrides
    def init(self, data: torch.Tensor, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, in_channels, H, W]
        z1, z2 = data.chunk(2, dim=1)
        mu = self.net.init(z1)
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            z2 = z2.mul(log_scale.exp())
            logdet = log_scale.view(z1.size(0), -1).sum(dim=1)
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
