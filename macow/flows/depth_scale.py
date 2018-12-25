__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.actnorm import ActNorm2dFlow
from macow.flows.conv import Conv1x1Flow
from macow.flows.macow import MaCowUnit


class DepthScaleStep(Flow):
    """
    A step of Macow Flows with 4 Macow Unit and a Glow step
    """
    def __init__(self, in_channels, kernel_size, scale=True, inverse=False):
        super(DepthScaleStep, self).__init__(inverse)
        num_units = 4
        units = [MaCowUnit(in_channels, kernel_size, scale=scale, inverse=inverse) for _ in range(num_units)]
        self.units = nn.ModuleList(units)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
        self.conv1x1 = Conv1x1Flow(in_channels, inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for unit in self.units:
            out, logdet = unit.forward(out, h=h)
            logdet_accum = logdet_accum + logdet

        out, logdet = self.actnorm.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        out, logdet = self.conv1x1.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.conv1x1.backward(input, h=h)
        out, logdet = self.actnorm.backward(out, h=h)
        logdet_accum = logdet_accum + logdet

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

        out, logdet = self.actnorm.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        out, logdet = self.conv1x1.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class DepthScaleBlock(Flow):
    """
    Bottom layer of depth upscale model.
    """
    def __init__(self, num_steps, in_channels, kernel_size, scale=True, inverse=False):
        super(DepthScaleBlock, self).__init__(inverse)
        steps = [DepthScaleStep(in_channels, kernel_size, scale=scale, inverse=inverse) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
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
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "DepthScaleBlock":
        return DepthScaleBlock(**params)
