__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.utils import _pair

from macow.flows.flow import Flow
from macow.utils import norm
from macow.nnet.weight_norm import MaskedConv2d, Conv2dWeightNorm


class Conv1x1Flow(Flow):
    def __init__(self, in_channels, inverse=False):
        super(Conv1x1Flow, self).__init__(inverse)
        self.in_channels = in_channels
        self.weight = Parameter(torch.Tensor(in_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)

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
        batch, channels, H, W = input.size()
        out = F.conv2d(input, self.weight.view(self.in_channels, self.in_channels, 1, 1))
        _, logdet = torch.slogdet(self.weight)
        return out, logdet * H * W

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
        batch, channels, H, W = input.size()
        out = F.conv2d(input, self.weight.inverse().view(self.in_channels, self.in_channels, 1, 1))
        _, logdet = torch.slogdet(self.weight)
        return out, logdet * H * W * -1.0

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}'.format(self.inverse, self.in_channels)

    @classmethod
    def from_params(cls, params: Dict) -> "Conv1x1Flow":
        return Conv1x1Flow(**params)


class Conv1x1WeightNormFlow(Flow):
    def __init__(self, in_channels, inverse=False):
        super(Conv1x1WeightNormFlow, self).__init__(inverse)
        self.in_channels = in_channels
        self.weight_v = Parameter(torch.Tensor(in_channels, in_channels))
        self.weight_g = Parameter(torch.Tensor(in_channels, 1))
        self.bias = Parameter(torch.Tensor(in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_v)
        _norm = norm(self.weight_v, 0).data + 1e-8
        self.weight_g.data.copy_(_norm.log())
        nn.init.constant_(self.bias, 0.)

    def compute_weight(self) -> torch.Tensor:
        _norm = norm(self.weight_v, 0) + 1e-8
        weight = self.weight_v * (self.weight_g.exp() / _norm)
        return weight

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
        batch, channels, H, W = input.size()
        weight = self.compute_weight()
        out = F.conv2d(input, weight.view(self.in_channels, self.in_channels, 1, 1), self.bias)
        _, logdet = torch.slogdet(weight)
        return out, logdet * H * W

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
        batch, channels, H, W = input.size()
        weight = self.compute_weight()
        out = F.conv2d(input - self.bias.view(self.in_channels, 1, 1), weight.inverse().view(self.in_channels, self.in_channels, 1, 1))
        _, logdet = torch.slogdet(weight)
        return out, logdet * H * W * -1.0

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out, _ = self.forward(data)
            out = out.transpose(0, 1).contiguous().view(self.in_channels, -1)
            # [n_channels]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-6)
            self.weight_g.add_(inv_stdv.log().unsqueeze(1))
            self.bias.add_(-mean).mul_(inv_stdv)
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}'.format(self.inverse, self.in_channels)

    @classmethod
    def from_params(cls, params: Dict) -> "Conv1x1WeightNormFlow":
        return Conv1x1WeightNormFlow(**params)


class MaskedConvFlow(Flow):
    """
    Masked Convolutional Flow
    """

    def __init__(self, in_channels, kernel_size, hidden_channels=None, order='A', scale=True, inverse=False):
        super(MaskedConvFlow, self).__init__(inverse)
        assert order in {'A', 'B'}, 'unknown order: {}'.format(order)
        self.in_channels = in_channels
        self.scale = scale
        if hidden_channels is None:
            hidden_channels = min(2 * in_channels, 512)
        out_channels = in_channels
        if scale:
            out_channels = out_channels * 2
        self.kernel_size = _pair(kernel_size)
        self.order = order
        self.net = nn.Sequential(
            MaskedConv2d(in_channels, hidden_channels, kernel_size, order=order),
            nn.ELU(inplace=True),
            Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=1, bias=True)
        )

    def init_net(self, x, h=None, init_scale=1.0):
        out = self.net[0].init(x, init_scale=init_scale)
        out = self.net[1](out)
        mu = self.net[2].init(out, init_scale=0.0)
        scale = None
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            scale = log_scale.add_(2.).sigmoid_()
        return mu, scale

    def calc_mu_and_scale(self, input: torch.Tensor, h=None):
        mu = self.net(input)
        scale = None
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            scale = log_scale.add_(2.).sigmoid_()
        return mu, scale

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
        mu, scale = self.calc_mu_and_scale(input, h=h)
        out = input
        if self.scale:
            out = out.mul(scale)
            logdet = scale.log().view(mu.size(0), -1).sum(dim=1)
        else:
            logdet = mu.new_zeros(mu.size(0))
        out = out + mu
        return out, logdet

    def backward_A(self, input: torch.Tensor, h=None) -> torch.Tensor:
        batch, channels, H, W = input.size()
        out = input.new_zeros(batch, channels, H, W)

        kH, kW = self.kernel_size
        cH = kH // 2
        cW = kW // 2
        for i in range(H):
            si = max(0, i - cH)
            for j in range(W):
                sj = max(0, j - cW)
                tj = min(W, j + cW + 1)
                input_curr = input[:, :, si:i+1, sj:tj]
                out_curr = out[:, :, si:i+1, sj:tj]
                # [batch, channels, cH, cW]
                mu, scale = self.calc_mu_and_scale(out_curr, h=h)
                new_out = input_curr - mu
                if self.scale:
                    new_out = new_out.div(scale + 1e-12)
                out[:, :, i, j] = new_out[:, :, -1, j - sj]
        return out

    def backward_B(self, input: torch.Tensor, h=None) -> torch.Tensor:
        batch, channels, H, W = input.size()
        out = input.new_zeros(batch, channels, H, W)

        kH, kW = self.kernel_size
        cH = kH // 2
        cW = kW // 2
        for i in reversed(range(H)):
            si = min(H, i + cH + 1)
            for j in reversed(range(W)):
                sj = max(0, j - cW)
                tj = min(W, j + cW + 1)
                input_curr = input[:, :, i:si, sj:tj]
                out_curr = out[:, :, i:si, sj:tj]
                # [batch, channels, cH, cW]
                mu, scale = self.calc_mu_and_scale(out_curr, h=h)
                new_out = input_curr - mu
                if self.scale:
                    new_out = new_out.div(scale + 1e-12)
                out[:, :, i, j] = new_out[:, :, 0, j - sj]
        return out

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
        if self.order == 'A':
            out = self.backward_A(input, h=h)
        else:
            out = self.backward_B(input, h=h)
        _, logdet = self.forward(out, h=h)
        return out, logdet.mul(-1.0)

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, scale = self.init_net(data, h=h, init_scale=init_scale)
        out = data
        if self.scale:
            out = out.mul(scale)
            logdet = scale.log().view(mu.size(0), -1).sum(dim=1)
        else:
            logdet = mu.new_zeros(mu.size(0))
        out = out + mu
        return out, logdet

    @classmethod
    def from_params(cls, params: Dict) -> "MaskedConvFlow":
        return MaskedConvFlow(**params)


Conv1x1Flow.register('conv1x1')
Conv1x1WeightNormFlow.register('conv1x1_weightnorm')
MaskedConvFlow.register('masked_conv')
