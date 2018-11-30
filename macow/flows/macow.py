__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.actnorm import ActNorm2dFlow
from macow.flows.conv import MaskedConvFlow, Conv1x1Flow
from macow.flows.nice import NICE
from macow.utils import squeeze2d, unsqueeze2d, split2d, unsplit2d
from macow.flows.glow import GlowStep


class MaCowUnit(Flow):
    """
    A Unit of Flows with an MCF(A), MCF(B), an Conv1x1, followd by an ActNorm and an activation.
    """
    def __init__(self, in_channels, kernel_size, scale=True, inverse=False):
        super(MaCowUnit, self).__init__(inverse)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
        self.conv1 = MaskedConvFlow(in_channels, kernel_size, order='A', scale=scale, inverse=inverse)
        self.conv2 = MaskedConvFlow(in_channels, kernel_size, order='B', scale=scale, inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # ActNorm
        out, logdet_accum = self.actnorm.forward(input, h=h)
        # MCF1
        out, logdet = self.conv1.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # MCF2
        out, logdet = self.conv2.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # MCF2
        out, logdet_accum = self.conv2.backward(input, h=h)
        # MCF1
        out, logdet = self.conv1.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        # ActNorm
        out, logdet = self.actnorm.backward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # ActNorm
        out, logdet_accum = self.actnorm.init(data, h=h, init_scale=init_scale)
        # MCF1
        out, logdet = self.conv1.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # MCF2
        out, logdet = self.conv2.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCowUnit":
        return MaCowUnit(**params)


class MaCowStep(Flow):
    """
    A step of Macow Flows with 4 Macow Unit and a Glow step
    """
    def __init__(self, in_channels, kernel_size, hidden_channels, scale=True, inverse=False, dropout=0.0):
        super(MaCowStep, self).__init__(inverse)
        num_units = 4
        units = [MaCowUnit(in_channels, kernel_size, scale=scale, inverse=inverse) for _ in range(num_units)]
        self.units = nn.ModuleList(units)
        self.glow_step = GlowStep(in_channels, hidden_channels=hidden_channels, scale=scale, inverse=inverse, dropout=dropout)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for unit in self.units:
            out, logdet = unit.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.glow_step.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.glow_step.backward(input, h=h)
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
        out, logdet = self.glow_step.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCowBottomBlock(Flow):
    """
    Masked Convolutional Flow Block (no squeeze nor split)
    """
    def __init__(self, num_steps, in_channels, kernel_size, inverse=False):
        super(MaCowBottomBlock, self).__init__(inverse)
        steps = [MaCowStep(in_channels, kernel_size, None, scale=False, inverse=inverse) for _ in range(num_steps)]
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


class MaCowTopBlock(Flow):
    """
    Masked Convolutional Flow Block (squeeze at beginning)
    """
    def __init__(self, num_steps, in_channels, kernel_size, scale=True, inverse=False, dropout=0.0):
        super(MaCowTopBlock, self).__init__(inverse)
        hidden_channels = 512
        steps = [MaCowStep(in_channels, kernel_size, hidden_channels, scale=scale, inverse=inverse, dropout=dropout) for _ in range(num_steps)]
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


class MaCowInternalBlock(Flow):
    """
    Masked Convolution Flow Internal Block (squeeze at beginning and split at end)
    """
    def __init__(self, num_steps, in_channels, kernel_size, hidden_channels, scale=True, inverse=False, dropout=0.0):
        super(MaCowInternalBlock, self).__init__(inverse)
        steps = [MaCowStep(in_channels, kernel_size, hidden_channels, scale=scale, inverse=inverse, dropout=dropout) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)
        self.prior = NICE(in_channels, hidden_channels=hidden_channels, scale=True, inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.prior.forward(out, h=h)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch]
        out, logdet_accum = self.prior.backward(input, h=h)
        for step in reversed(self.steps):
            out, logdet = step.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.prior.init(out, h=h, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCow(Flow):
    """
    Masked Convolutional Flow
    """
    def __init__(self, levels, num_steps, in_channels, kernel_size, scale=True, inverse=False, dropout=0.0):
        super(MaCow, self).__init__(inverse)
        assert levels > 1, 'MaCow should have at least 2 levels.'
        assert levels == len(num_steps)
        blocks = []
        self.levels = levels
        for level in range(levels):
            if level == 0:
                macow_block = MaCowBottomBlock(num_steps[level], in_channels, kernel_size, inverse=inverse)
                blocks.append(macow_block)
            elif level == levels - 1:
                in_channels = in_channels * 4
                macow_block = MaCowTopBlock(num_steps[level], in_channels, kernel_size, scale=scale, inverse=inverse, dropout=dropout)
                blocks.append(macow_block)
            else:
                in_channels = in_channels * 4
                # half = levels / 2
                # hidden_channels = 256 if level < half else 512
                hidden_channels = 512
                macow_block = MaCowInternalBlock(num_steps[level], in_channels, kernel_size, hidden_channels, scale=scale, inverse=inverse, dropout=dropout)
                blocks.append(macow_block)
                in_channels = in_channels // 2
        self.blocks = nn.ModuleList(blocks)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        outputs = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, MaCowInternalBlock) or isinstance(block, MaCowTopBlock):
                out = squeeze2d(out, factor=2)
            out, logdet = block.forward(out, h=h)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock):
                out1, out2 = split2d(out)
                outputs.append(out2)
                out = out1

        out = unsqueeze2d(out, factor=2)
        for _ in range(self.levels - 2):
            out2 = outputs.pop()
            out = unsqueeze2d(unsplit2d(out, out2), factor=2)
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        out = squeeze2d(input, factor=2)
        for _ in range(self.levels - 2):
            out1, out2 = split2d(out)
            outputs.append(out2)
            out = squeeze2d(out1, factor=2)

        logdet_accum = input.new_zeros(input.size(0))
        for i, block in enumerate(reversed(self.blocks)):
            if isinstance(block, MaCowInternalBlock):
                out2 = outputs.pop()
                out = unsplit2d(out, out2)
            out, logdet = block.backward(out, h=h)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock) or isinstance(block, MaCowTopBlock):
                out = unsqueeze2d(out, factor=2)
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        outputs = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, MaCowInternalBlock) or isinstance(block, MaCowTopBlock):
                out = squeeze2d(out, factor=2)
            out, logdet = block.init(out, h=h, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock):
                out1, out2 = split2d(out)
                outputs.append(out2)
                out = out1

        out = unsqueeze2d(out, factor=2)
        for _ in range(self.levels - 2):
            out2 = outputs.pop()
            out = unsqueeze2d(unsplit2d(out, out2), factor=2)
        assert len(outputs) == 0
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCow":
        return MaCow(**params)


MaCow.register('macow')
