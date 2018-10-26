__author__ = 'max'

import warnings
from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.conv import MaskedConvFlow, Conv1x1Flow
from macow.flows.activation import IdentityFlow
from macow.utils import squeeze2d, unsqueeze2d


class MaCowUnit(Flow):
    """
    A Unit of Flows with an MCF(A), MCF(B), an Conv1x1, followd by an ActNorm and an activation.
    """
    def __init__(self, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowUnit, self).__init__(inverse)
        self.conv1 = MaskedConvFlow(in_channels, kernel_size, mask_type='A', inverse=inverse)
        self.conv2 = MaskedConvFlow(in_channels, kernel_size, mask_type='B', inverse=inverse)
        self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse)
        if activation.inverse != inverse:
            activation.inverse = inverse
            warnings.warn('activation inverse does not match MaCow inverse')
        self.activation = activation

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.conv1.forward(input, h=h)
        out, logdet = self.conv2.forward(out, h=h)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.conv1x1.forward(out, h=h)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.activation.forward(out)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.activation.backward(input)

        out, logdet = self.conv1x1.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.conv2.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        out, logdet = self.conv1.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.conv1.init(data, h=h, init_scale=init_scale)
        out, logdet = self.conv2.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.conv1x1.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.activation.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCowUnit":
        activation_params = params.pop('activation')
        activation = Flow.by_name(activation_params.pop('type')).from_params(activation_params)
        return MaCowUnit(**params, activation=activation)

class MaCowBlock(Flow):
    """
    Masked Convolutional Flow Block
    """
    def __init__(self, num_units, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowBlock, self).__init__(inverse)
        units = [MaCowUnit(in_channels, kernel_size, activation, inverse=inverse) for _ in range(num_units)]
        self.units = nn.ModuleList(units)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for unit in self.units:
            out, logdet = unit.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for unit in reversed(self.units):
            out, logdet = unit.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        for unit in self.units:
            out, logdet = unit.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCow(Flow):
    """
    Masked Convolutional Flow
    """
    def __init__(self, levels, num_units, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCow, self).__init__(inverse)
        assert levels == len(num_units)
        blocks = []
        self.levels = levels
        for level in range(levels):
            macow_block = MaCowBlock(num_units[level], in_channels, kernel_size, activation, inverse=inverse)
            blocks.append(macow_block)
            in_channels = in_channels * 4
        self.blocks = nn.ModuleList(blocks)
        in_channels = in_channels // 4
        self.output_unit = MaCowUnit(in_channels, kernel_size, IdentityFlow(inverse), inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for i, block in enumerate(self.blocks):
            if i > 0:
                out = squeeze2d(out, factor=2)
            out, logdet = block.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.output_unit.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # unsqueeze
        for i in range(self.levels - 1):
            out = unsqueeze2d(out, factor=2)
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # squeeze first
        for i in range(self.levels - 1):
            out = squeeze2d(out, factor=2)
        out, logdet_accum = self.output_unit.backward(out, h=h)
        for i, block in enumerate(reversed(self.blocks)):
            if i > 0:
                out = unsqueeze2d(out, factor=2)
            out, logdet = block.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        for i, block in enumerate(self.blocks):
            if i > 0:
                out = squeeze2d(out, factor=2)
            out, logdet = block.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.output_unit.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # unsqueeze
        for i in range(self.levels - 1):
            out = unsqueeze2d(out, factor=2)
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCow":
        activation_params = params.pop('activation')
        activation = Flow.by_name(activation_params.pop('type')).from_params(activation_params)
        return MaCow(**params, activation=activation)


MaCow.register('macow')
