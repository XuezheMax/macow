__author__ = 'max'

import warnings
from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.conv import MaskedConvFlow, Conv1x1Flow
from macow.flows.actnorm import ActNorm2dFlow


class MaCowBlock(Flow):
    """
    A Block of Flows with an MCF(A), MCF(B), an Conv1x1, followd by an ActNorm and an activation.
    """
    def __init__(self, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowBlock, self).__init__(inverse)
        self.conv1 = MaskedConvFlow(in_channels, kernel_size, mask_type='A', inverse=inverse)
        self.conv2 = MaskedConvFlow(in_channels, kernel_size, mask_type='B', inverse=inverse)
        self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
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

        out, logdet = self.actnorm.forward(out, h=h)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.activation.forward(out)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.activation.backward(input)

        out, logdet = self.actnorm.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

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

        out, logdet = self.actnorm.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        out, logdet = self.activation.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet

        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCowBlock":
        activation_params = params.pop('activation')
        activation = Flow.by_name(activation_params.pop('type')).from_params(activation_params)
        return MaCowBlock(**params, activation=activation)

class MaCow(Flow):
    """
    Masked Convolutional Flow
    """
    def __init__(self, num_blocks, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCow, self).__init__(inverse)
        blocks = [MaCowBlock(in_channels, kernel_size, activation, inverse=inverse) for _ in range(num_blocks)]
        self.blocks = nn.ModuleList(blocks)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for block in self.blocks:
            out, logdet = block.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for block in reversed(self.blocks):
            out, logdet = block.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        for block in self.blocks:
            out, logdet = block.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCow":
        activation_params = params.pop('activation')
        activation = Flow.by_name(activation_params.pop('type')).from_params(activation_params)
        return MaCow(**params, activation=activation)


MaCow.register('macow')
