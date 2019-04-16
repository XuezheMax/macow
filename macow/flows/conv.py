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
from macow.nnet.weight_norm import Conv2dWeightNorm, ShiftedConv2d


class Conv1x1Flow(Flow):
    def __init__(self, in_channels, inverse=False):
        super(Conv1x1Flow, self).__init__(inverse)
        self.in_channels = in_channels
        self.weight = Parameter(torch.Tensor(in_channels, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)

    @overrides
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        batch, channels, H, W = input.size()
        out = F.conv2d(input, self.weight.view(self.in_channels, self.in_channels, 1, 1))
        _, logdet = torch.slogdet(self.weight)
        return out, logdet.mul(H * W)

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        batch, channels, H, W = input.size()
        out = F.conv2d(input, self.weight.inverse().view(self.in_channels, self.in_channels, 1, 1))
        _, logdet = torch.slogdet(self.weight)
        return out, logdet.mul(H * -W)

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
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
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        batch, channels, H, W = input.size()
        weight = self.compute_weight()
        out = F.conv2d(input, weight.view(self.in_channels, self.in_channels, 1, 1), self.bias)
        _, logdet = torch.slogdet(weight)
        return out, logdet.mul(H * W)

    @overrides
    def backward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`
        """
        batch, channels, H, W = input.size()
        weight = self.compute_weight()
        out = F.conv2d(input - self.bias.view(self.in_channels, 1, 1), weight.inverse().view(self.in_channels, self.in_channels, 1, 1))
        _, logdet = torch.slogdet(weight)
        return out, logdet.mul(H * -W)

    @overrides
    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
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


class MCFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_channels, order):
        super(MCFBlock, self).__init__()
        self.masked_conv = ShiftedConv2d(in_channels, hidden_channels, kernel_size, order=order, bias=True)
        self.conv1x1 = Conv2dWeightNorm(hidden_channels, out_channels, kernel_size=1, bias=True)
        self.activation = nn.ELU(inplace=True)

    def forward(self, x, s=None, shifted=True):
        c = self.masked_conv(x, shifted=shifted)
        if s is not None:
            c = c + s
        c = self.conv1x1(self.activation(c))
        return c

    def init(self, x, s=None, init_scale=1.0):
        c = self.masked_conv.init(x, init_scale=init_scale)
        if s is not None:
            c = c + s
        c = self.conv1x1.init(self.activation(c), init_scale=0.0)
        return c


class MaskedConvFlow(Flow):
    """
    Masked Convolutional Flow
    """

    def __init__(self, in_channels, kernel_size, hidden_channels=None, s_channels=None, order='A', scale=True, inverse=False):
        super(MaskedConvFlow, self).__init__(inverse)
        self.in_channels = in_channels
        self.scale = scale
        if hidden_channels is None:
            if in_channels <= 96:
                hidden_channels = 4 * in_channels
            else:
                hidden_channels = min(2 * in_channels, 512)
        out_channels = in_channels
        if scale:
            out_channels = out_channels * 2
        self.kernel_size = kernel_size
        self.order = order
        self.net = MCFBlock(in_channels, out_channels, kernel_size, hidden_channels, order)
        if s_channels is None or s_channels <= 0:
            self.s_conv = None
        else:
            self.s_conv = Conv2dWeightNorm(s_channels, hidden_channels, (3, 3), bias=True, padding=1)

    def calc_mu_and_scale(self, x: torch.Tensor, s=None, shifted=True):
        mu = self.net(x, s=s, shifted=shifted)
        scale = None
        if self.scale:
            mu, log_scale = mu.chunk(2, dim=1)
            scale = log_scale.add_(2.).sigmoid_()
        return mu, scale

    def init_net(self, x, s=None, init_scale=1.0):
        mu = self.net.init(x, s=s, init_scale=init_scale)
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
        if self.s_conv is not None:
            s = self.s_conv(s)
        mu, scale = self.calc_mu_and_scale(input, s=s)
        out = input
        if self.scale:
            out = out.mul(scale)
            logdet = scale.log().view(mu.size(0), -1).sum(dim=1)
        else:
            logdet = mu.new_zeros(mu.size(0))
        out = out + mu
        return out, logdet

    def backward_height(self, input: torch.Tensor, s=None, reverse=False) -> torch.Tensor:
        batch, channels, H, W = input.size()

        kH, kW = self.kernel_size
        cW = kW // 2
        out = input.new_zeros(batch, channels, H + kH, W + 2 * cW)

        itr = reversed(range(H)) if reverse else range(H)
        for h in itr:
            curr_h = h if reverse else h + kH
            s_h = h + 1 if reverse else h
            t_h = h + kH + 1 if reverse else h + kH
            # [batch, channels, kH, width+2*cW]
            out_curr = out[:, :, s_h:t_h]
            s_curr = None if s is None else s[:, :, h:h + 1]
            # [batch, channels, width]
            in_curr = input[:, :, h]

            # [batch, channels, 1, width]
            mu, scale = self.calc_mu_and_scale(out_curr, s=s_curr, shifted=False)
            # [batch, channels, width]
            new_out = in_curr - mu.squeeze(2)
            if self.scale:
                new_out = new_out.div(scale.squeeze(2) + 1e-12)
            out[:, :, curr_h, cW:W + cW] = new_out

        out = out[:, :, :H, cW:cW + W] if reverse else out[:, :, kH:, cW:cW + W]
        return out

    def backward_width(self, input: torch.Tensor, s=None, reverse=False) -> torch.Tensor:
        batch, channels, H, W = input.size()

        kH, kW = self.kernel_size
        cH = kH // 2
        out = input.new_zeros(batch, channels, H + 2 * cH, W + kW)

        itr = reversed(range(W)) if reverse else range(W)
        for w in itr:
            curr_w = w if reverse else w + kW
            s_w = w + 1 if reverse else w
            t_w = w + kW + 1 if reverse else w + kW
            # [batch, channels, height+2*cH, kW]
            out_curr = out[:, :, :, s_w:t_w]
            s_curr = None if s is None else s[:, :, :, w:w + 1]
            # [batch, channels, height]
            in_curr = input[:, :, :, w]

            # [batch, channels, height, 1]
            mu, scale = self.calc_mu_and_scale(out_curr, s=s_curr, shifted=False)
            # [batch, channels, height]
            new_out = in_curr - mu.squeeze(3)
            if self.scale:
                new_out = new_out.div(scale.squeeze(3) + 1e-12)
            out[:, :, cH:H + cH, curr_w] = new_out

        out = out[:, :, cH:cH + H, :W] if reverse else out[:, :, cH:cH + H, kW:]
        return out

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
        if self.s_conv is not None:
            ss = self.s_conv(s)
        else:
            ss = s
        if self.order == 'A':
            out = self.backward_height(input, s=ss, reverse=False)
        elif self.order == 'B':
            out = self.backward_height(input, s=ss, reverse=True)
        elif self.order == 'C':
            out = self.backward_width(input, s=ss, reverse=False)
        else:
            out = self.backward_width(input, s=ss, reverse=True)
        _, logdet = self.forward(out, s=s)
        return out, logdet.mul(-1.0)

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.s_conv is not None:
            s = self.s_conv.init(s, init_scale=init_scale)
        mu, scale = self.init_net(data, s=s, init_scale=init_scale)
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
