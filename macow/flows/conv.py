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

    def __init__(self, in_channels, kernel_size, mask_type='A', inverse=False):
        super(MaskedConvFlow, self).__init__(inverse)
        assert mask_type in {'A', 'B'}, 'unknown mask type: {}'.format(mask_type)
        self.in_channels = in_channels
        self.kernel_size = _pair(kernel_size)
        self.mask_type = mask_type
        for k in self.kernel_size:
            assert k % 2 == 1, 'kernel cannot include even number: {}'.format(self.kernel_size)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        self.weight_v = Parameter(torch.Tensor(in_channels, 1, *self.kernel_size))
        self.weight_g = Parameter(torch.Tensor(in_channels, 1, 1, 1))
        self.bias = Parameter(torch.Tensor(in_channels))
        self.register_buffer('mask', torch.ones(self.kernel_size))
        kH, kW = self.kernel_size
        mask = np.ones([*self.mask.size()], dtype=np.float32)
        mask[kH // 2, kW // 2 + 1:] = 0
        mask[kH // 2 + 1:] = 0
        # reverse order
        if self.mask_type == 'B':
            reverse_mask = mask[::-1, :]
            reverse_mask = reverse_mask[:, ::-1]
            mask = reverse_mask.copy()
        self.mask.copy_(torch.from_numpy(mask).float())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_v, mean=0.0, std=0.05)
        with torch.no_grad():
            cH = self.kernel_size[0] // 2
            cW = self.kernel_size[1] // 2
            self.weight_v[:, 0, cH, cW].add_(1.0)
        self.weight_v.data.mul_(self.mask)
        _norm = norm(self.weight_v, 0).data + 1e-8
        self.weight_g.data.copy_(_norm.log())
        nn.init.constant_(self.bias, 0)

    def compute_weight(self) -> torch.Tensor:
        self.weight_v.data.mul_(self.mask)
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
        # [batch, in_channels, H, W]
        out = F.conv2d(input, weight, bias=self.bias, padding=self.padding, groups=channels)

        cH = self.kernel_size[0] // 2
        cW = self.kernel_size[1] // 2
        # [in_channels]
        logdet = weight[:, 0, cH, cW]
        logdet = logdet.abs().log().sum() * H * W
        return out, logdet

    def backward_A(self, input: torch.Tensor, H, W, c_weight: torch.Tensor, h=None) -> torch.Tensor:
        kH, kW = self.kernel_size
        cH = kH // 2
        cW = kW // 2
        out = input.new_zeros(input.size())
        for i in range(H):
            si = max(0, i - cH)
            for j in range(W):
                sj = max(0, j - cW)
                tj = min(W, j + cW + 1)
                input_curr = input[:, :, si:i+1, sj:tj]
                out_curr = out[:, :, si:i+1, sj:tj]
                # [batch, channels, cH, cW]
                tmp, _ = self.forward(out_curr, h=h)
                new_out = (input_curr - tmp).div(c_weight)
                out[:, :, i, j] = new_out[:, :, -1, j - sj]
        return out

    def backward_B(self, input: torch.Tensor, H, W, c_weight: torch.Tensor, h=None) -> torch.Tensor:
        kH, kW = self.kernel_size
        cH = kH // 2
        cW = kW // 2
        out = input.new_zeros(input.size())
        for i in reversed(range(H)):
            si = min(H, i + cH + 1)
            for j in reversed(range(W)):
                sj = max(0, j - cW)
                tj = min(W, j + cW + 1)
                input_curr = input[:, :, i:si, sj:tj]
                out_curr = out[:, :, i:si, sj:tj]
                # [batch, channels, cH, cW]
                tmp, _ = self.forward(out_curr, h=h)
                new_out = (input_curr - tmp).div(c_weight)
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
        kH, kW = self.kernel_size
        cH = kH // 2
        cW = kW // 2
        batch, channels, H, W = input.size()

        weight = self.compute_weight()
        c_weight = weight[:, 0, cH, cW]
        logdet = c_weight.abs().log().sum() * H * W * -1.0

        # [channels, 1, 1]
        c_weight = c_weight.unsqueeze(1).unsqueeze(1)
        if self.mask_type == 'A':
            out = self.backward_A(input, H, W, c_weight, h=h)
        else:
            out = self.backward_B(input, H, W, c_weight, h=h)
        return out, logdet

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out, _ = self.forward(data, h=h)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            # [n_channels]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-6)
            self.weight_g.add_(inv_stdv.log().view(n_channels, 1, 1, 1))
            self.bias.add_(-mean).mul_(inv_stdv)
            return self.forward(data, h=h)

    @overrides
    def extra_repr(self):
        return 'inverse={}, type={}, in_channels={}, kernel_size={}, padding={}'.format(self.inverse, self.mask_type,
                                                                                        self.in_channels, self.kernel_size, self.padding)

    @classmethod
    def from_params(cls, params: Dict) -> "MaskedConvFlow":
        return MaskedConvFlow(**params)


Conv1x1Flow.register('conv1x1')
Conv1x1WeightNormFlow.register('conv1x1_weightnorm')
MaskedConvFlow.register('masked_conv')
