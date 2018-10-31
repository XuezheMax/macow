__author__ = 'max'

import warnings
from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.conv import MaskedConvFlow, Conv1x1Flow
from macow.flows.activation import IdentityFlow
from macow.utils import squeeze2d, unsqueeze2d, split2d, unsplit2d


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

class MaCowTopBlock(Flow):
    """
    Masked Convolutional Flow Block (No squeeze and split)
    """
    def __init__(self, num_units, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowTopBlock, self).__init__(inverse)
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


class MaCowBottomBlock(Flow):
    """
    Masked Convolutional Flow Block (squeeze at beginning)
    """
    def __init__(self, num_units, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowBottomBlock, self).__init__(inverse)
        units = [MaCowUnit(in_channels, kernel_size, activation, inverse=inverse) for _ in range(num_units)]
        self.units = nn.ModuleList(units)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, channels*factor*factor, height/factor, width/factor]
        out = squeeze2d(input, factor=2)
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
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
        out = unsqueeze2d(out, factor=2)
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, channels*factor*factor, height/factor, width/factor]
        out = squeeze2d(data, factor=2)
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for unit in self.units:
            out, logdet = unit.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCowInternalBlock(Flow):
    """
    Masked Convolution Flow Internal Block (squeeze at beginning and split at end)
    """
    def __init__(self, num_units, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowInternalBlock, self).__init__(inverse)
        units = [MaCowUnit(in_channels, kernel_size, activation, inverse=inverse) for _ in range(num_units)]
        self.units = nn.ModuleList(units)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, channels*factor*factor, height/factor, width/factor]
        out = squeeze2d(input, factor=2)
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for unit in self.units:
            out, logdet = unit.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        # [2*batch, channels*factor*factor/2, height/factor, width/factor]
        out = split2d(out)
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch / 2, channels * 2, height, width]
        out = unsplit2d(input)
        logdet_accum = input.new_zeros(out.size(0))
        for unit in reversed(self.units):
            out, logdet = unit.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        out = unsqueeze2d(out, factor=2)
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, channels*factor*factor, height/factor, width/factor]
        out = squeeze2d(data, factor=2)
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for unit in self.units:
            out, logdet = unit.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        # [2*batch, channels*factor*factor/2, height/factor, width/factor]
        out = split2d(out)
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
            if level == 0:
                macow_block = MaCowTopBlock(num_units[level], in_channels, kernel_size, activation, inverse=inverse)
                blocks.append(macow_block)
            elif level == levels - 1:
                macow_block = MaCowBottomBlock(num_units[level], in_channels * 4, kernel_size, activation, inverse=inverse)
                blocks.append(macow_block)
                in_channels = in_channels * 4
            else:
                macow_block = MaCowInternalBlock(num_units[level], in_channels * 4, kernel_size, activation, inverse=inverse)
                blocks.append(macow_block)
                in_channels = in_channels * 2
        self.blocks = nn.ModuleList(blocks)
        self.output_unit = MaCowUnit(in_channels, kernel_size, IdentityFlow(inverse), inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        factor = 1
        for i, block in enumerate(self.blocks):
            out, logdet = block.forward(out, h=h)
            # [factor * batch] -> [batch]
            logdet = sum(logdet.chunk(factor, dim=0))
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock):
                factor = factor * 2

        # output unit
        out, logdet = self.output_unit.forward(out, h=h)
        # [factor * batch] -> [batch]
        logdet = sum(logdet.chunk(factor, dim=0))
        logdet_accum = logdet_accum + logdet

        # unsqueeze & unsplit
        if self.levels > 1:
            out = unsqueeze2d(out, factor=2)
        for i in range(self.levels - 2):
            out = unsqueeze2d(unsplit2d(out), factor=2)
            factor = factor // 2
        assert factor == 1
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # squeeze & split first
        factor = 1
        for i in range(self.levels - 2):
            factor = factor * 2
            out = split2d(squeeze2d(out, factor=2))
        if self.levels > 1:
            out = squeeze2d(out)

        # output unit
        out, logdet_accum = self.output_unit.backward(out, h=h)
        # [factor * batch] -> [batch]
        logdet_accum = sum(logdet_accum.chunk(factor, dim=0))

        for i, block in enumerate(reversed(self.blocks)):
            if isinstance(block, MaCowInternalBlock):
                factor = factor // 2
            out, logdet = block.backward(out, h=h)
            # [factor * batch] -> [batch]
            logdet = sum(logdet.chunk(factor, dim=0))
            logdet_accum = logdet_accum + logdet
        assert factor == 1
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        factor = 1
        for i, block in enumerate(self.blocks):
            out, logdet = block.init(out, h=h, init_scale=init_scale)
            # [factor * batch] -> [batch]
            logdet = sum(logdet.chunk(factor, dim=0))
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock):
                factor = factor * 2

        # output unit
        out, logdet = self.output_unit.init(out, h=h, init_scale=init_scale)
        # [factor * batch] -> [batch]
        logdet = sum(logdet.chunk(factor, dim=0))
        logdet_accum = logdet_accum + logdet

        # unsqueeze & unsplit
        if self.levels > 1:
            out = unsqueeze2d(out, factor=2)
        for i in range(self.levels - 2):
            out = unsqueeze2d(unsplit2d(out), factor=2)
            factor = factor // 2
        assert factor == 1
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCow":
        activation_params = params.pop('activation')
        activation = Flow.by_name(activation_params.pop('type')).from_params(activation_params)
        return MaCow(**params, activation=activation)


MaCow.register('macow')
