__author__ = 'max'

import math
import warnings
from overrides import overrides
from typing import Dict, Tuple, List
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.conv import MaskedConvFlow, Conv1x1Flow
from macow.flows.activation import IdentityFlow
from macow.flows.nice import NICE
from macow.utils import squeeze2d, unsqueeze2d, split2d, unsplit2d
from macow.nnet import Conv2dWeightNorm


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


class MaCowStep(Flow):
    """
    A step of Macow Flows with 4 Macow Unit and a NICE coupling layer
    """
    def __init__(self, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowStep, self).__init__(inverse)
        num_units = 3
        units = [MaCowUnit(in_channels, kernel_size, activation, inverse=inverse) for _ in range(num_units)]
        units.append(MaCowUnit(in_channels, kernel_size, IdentityFlow(inverse=inverse), inverse=inverse))
        self.units = nn.ModuleList(units)
        self.coupling = NICE(in_channels, inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for unit in self.units:
            out, logdet = unit.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.coupling.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.coupling.backward(input, h=h)
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
        out, logdet = self.coupling.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

class MaCowBottomBlock(Flow):
    """
    Masked Convolutional Flow Block (No squeeze and split)
    """
    def __init__(self, num_units, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowBottomBlock, self).__init__(inverse)
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


class MaCowTopBlock(Flow):
    """
    Masked Convolutional Flow Block (squeeze at beginning)
    """
    def __init__(self, num_steps, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowTopBlock, self).__init__(inverse)
        steps = [MaCowStep(in_channels, kernel_size, activation, inverse=inverse) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, channels*factor*factor, height/factor, width/factor]
        out = squeeze2d(input, factor=2)
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for step in reversed(self.steps):
            out, logdet = step.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        out = unsqueeze2d(out, factor=2)
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, channels*factor*factor, height/factor, width/factor]
        out = squeeze2d(data, factor=2)
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCowInternalBlock(Flow):
    """
    Masked Convolution Flow Internal Block (squeeze at beginning and split at end)
    """
    def __init__(self, num_steps, in_channels, kernel_size, activation: Flow, inverse=False):
        super(MaCowInternalBlock, self).__init__(inverse)
        steps = [MaCowStep(in_channels, kernel_size, activation, inverse=inverse) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)
        self.prior = nn.Sequential(
            Conv2dWeightNorm(in_channels // 2, in_channels * 2, 3, padding=1),
            nn.ELU(),
            Conv2dWeightNorm(in_channels * 2, in_channels, 3, padding=1)
        )

    def init_prior(self, z, init_scale=1.0):
        out = z
        for layer in self.prior:
            if isinstance(layer, nn.ELU):
                out = layer(out)
            else:
                out = layer.init(out, init_scale=init_scale)
        return out.chunk(2, dim=1)

    def forward_prior(self, z):
        out = self.prior(z)
        return out.chunk(2, dim=1)

    def calc_prior_logp(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor):
        eps = 1e-12
        log_probs = logvar + (z - mu).pow(2).div(logvar.exp() + eps) + math.log(math.pi * 2.)
        log_probs = log_probs.view(z.size(0), -1).sum(dim=1) * -0.5
        return log_probs

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # [batch, channels*factor*factor, height/factor, width/factor]
        out = squeeze2d(input, factor=2)
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        # [batch, channels*factor*factor/2, height/factor, width/factor] * 2
        out1, out2 = split2d(out)
        mu, logvar = self.forward_prior(out1)
        std = logvar.mul(0.5).exp()
        eps = (out2 - mu).div(std + 1e-12)
        logp = self.calc_prior_logp(out2, mu, logvar)
        return (out1, eps), logdet_accum + logp

    @overrides
    def backward(self, input: torch.Tensor, eps=None, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, channels, height, width]
        mu, logvar = self.forward_prior(input)
        std = logvar.mul(0.5).exp()
        if eps is None:
            eps = input.new_empty(input.size()).normal_()
        out = eps.mul(std).add(mu)
        logp = self.calc_prior_logp(out, mu, logvar)
        # [batch, channels * 2, height, width]
        out = unsplit2d(input, out)

        logdet_accum = logp * -1.
        for step in reversed(self.steps):
            out, logdet = step.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        out = unsqueeze2d(out, factor=2)
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # [batch, channels*factor*factor, height/factor, width/factor]
        out = squeeze2d(data, factor=2)
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        # [batch, channels*factor*factor/2, height/factor, width/factor] * 2
        out1, out2 = split2d(out)
        mu, logvar = self.init_prior(out1, init_scale=init_scale)
        std = logvar.mul(0.5).exp()
        eps = (out2 - mu).div(std + 1e-12)
        logp = self.calc_prior_logp(out2, mu, logvar)
        return (out1, eps), logdet_accum + logp


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
                macow_block = MaCowBottomBlock(num_units[level], in_channels, kernel_size, activation, inverse=inverse)
                blocks.append(macow_block)
            elif level == levels - 1:
                macow_block = MaCowTopBlock(num_units[level], in_channels * 4, kernel_size, activation, inverse=inverse)
                blocks.append(macow_block)
                in_channels = in_channels * 4
            else:
                macow_block = MaCowInternalBlock(num_units[level], in_channels * 4, kernel_size, activation, inverse=inverse)
                blocks.append(macow_block)
                in_channels = in_channels * 2
        self.blocks = nn.ModuleList(blocks)
        self.output_unit = MaCowUnit(in_channels, kernel_size, IdentityFlow(inverse), inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        eps = []
        for i, block in enumerate(self.blocks):
            out, logdet = block.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock):
                out, e = out
                eps.append(e)
            else:
                eps.append(None)

        # output unit
        out, logdet = self.output_unit.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        assert len(eps) == self.levels
        return out, logdet_accum, eps

    @overrides
    def backward(self, input: torch.Tensor, eps=None, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # output unit
        out, logdet_accum = self.output_unit.backward(out, h=h)

        if eps is not None:
            eps = eps[::-1]
        else:
            eps = [None] * self.levels

        for i, block in enumerate(reversed(self.blocks)):
            if isinstance(block, MaCowInternalBlock):
                out, logdet = block.backward(out, eps=eps[i], h=h)
            else:
                out, logdet = block.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        eps = []
        for i, block in enumerate(self.blocks):
            out, logdet = block.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock):
                out, e = out
                eps.append(e)
            else:
                eps.append(None)

        # output unit
        out, logdet = self.output_unit.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum, eps

    @classmethod
    def from_params(cls, params: Dict) -> "MaCow":
        activation_params = params.pop('activation')
        activation = Flow.by_name(activation_params.pop('type')).from_params(activation_params)
        return MaCow(**params, activation=activation)


MaCow.register('macow')
