__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.actnorm import ActNorm2dFlow
from macow.flows.conv import MaskedConvFlow
from macow.flows.nice import NICE
from macow.utils import squeeze2d, unsqueeze2d, split2d, unsplit2d
from macow.flows.glow import GlowStep


class MaCowUnit(Flow):
    """
    A Unit of Flows with an MCF(A), MCF(B), an Conv1x1, followd by an ActNorm and an activation.
    """
    def __init__(self, in_channels, kernel_size, s_channels, scale=True, inverse=False):
        super(MaCowUnit, self).__init__(inverse)
        self.actnorm = ActNorm2dFlow(in_channels, inverse=inverse)
        self.conv1 = MaskedConvFlow(in_channels, kernel_size, s_channels=s_channels, order='A', scale=scale, inverse=inverse)
        self.conv2 = MaskedConvFlow(in_channels, kernel_size, s_channels=s_channels, order='B', scale=scale, inverse=inverse)

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # ActNorm
        out, logdet_accum = self.actnorm.forward(input)
        # MCF1
        out, logdet = self.conv1.forward(out, s=s)
        logdet_accum = logdet_accum + logdet
        # MCF2
        out, logdet = self.conv2.forward(out, s=s)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # MCF2
        out, logdet_accum = self.conv2.backward(input, s=s)
        # MCF1
        out, logdet = self.conv1.backward(out, s=s)
        logdet_accum = logdet_accum + logdet
        # ActNorm
        out, logdet = self.actnorm.backward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        # ActNorm
        out, logdet_accum = self.actnorm.init(data, init_scale=init_scale)
        # MCF1
        out, logdet = self.conv1.init(out, s=s, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        # MCF2
        out, logdet = self.conv2.init(out, s=s, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCowUnit":
        return MaCowUnit(**params)


class MaCowStep(Flow):
    """
    A step of Macow Flows with 4 Macow Unit and a Glow step
    """
    def __init__(self, in_channels, kernel_size, hidden_channels, s_channels, dilation, scale=True, inverse=False):
        super(MaCowStep, self).__init__(inverse)
        num_units = 4
        units = [MaCowUnit(in_channels, kernel_size, s_channels, scale=scale, inverse=inverse) for _ in range(num_units)]
        self.units = nn.ModuleList(units)
        self.glow_step = GlowStep(in_channels, hidden_channels=hidden_channels, s_channels=s_channels, scale=scale, inverse=inverse, dilation=dilation)

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for unit in self.units:
            out, logdet = unit.forward(out, s=s)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.glow_step.forward(out, s=s)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.glow_step.backward(input, s=s)
        for unit in reversed(self.units):
            out, logdet = unit.backward(out, s=s)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        for unit in self.units:
            out, logdet = unit.init(out, s=s, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        out, logdet = self.glow_step.init(out, s=s, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCowBottomBlock(Flow):
    """
    Masked Convolutional Flow Block (no squeeze nor split)
    """
    def __init__(self, num_steps, in_channels, kernel_size, s_channels, dilation, scale=False, inverse=False):
        super(MaCowBottomBlock, self).__init__(inverse)
        steps = [MaCowStep(in_channels, kernel_size, None, s_channels, dilation, scale=scale, inverse=inverse) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out, s=s)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for step in reversed(self.steps):
            out, logdet = step.backward(out, s=s)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, s=s, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCowTopBlock(Flow):
    """
    Masked Convolutional Flow Block (squeeze at beginning)
    """
    def __init__(self, num_steps, in_channels, kernel_size, s_channels, dilation, scale=True, inverse=False):
        super(MaCowTopBlock, self).__init__(inverse)
        hidden_channels = 512
        steps = [MaCowStep(in_channels, kernel_size, hidden_channels, s_channels, dilation, scale=scale, inverse=inverse) for _ in range(num_steps)]
        self.steps = nn.ModuleList(steps)

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        for step in self.steps:
            out, logdet = step.forward(out, s=s)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        for step in reversed(self.steps):
            out, logdet = step.backward(out, s=s)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        for step in self.steps:
            out, logdet = step.init(out, s=s, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
        return out, logdet_accum


class MaCowInternalBlock(Flow):
    """
    Masked Convolution Flow Internal Block (squeeze at beginning and split at end)
    """
    def __init__(self, num_steps, in_channels, kernel_size, hidden_channels, s_channels, dilation, factor=2, scale=True, inverse=False):
        super(MaCowInternalBlock, self).__init__(inverse)
        num_layers = len(num_steps)
        assert num_layers < factor
        self.layers = nn.ModuleList()
        self.priors = nn.ModuleList()
        channel_step = in_channels // factor
        for num_step in num_steps:
            layer = [MaCowStep(in_channels, kernel_size, hidden_channels, s_channels, dilation, scale=scale, inverse=inverse) for _ in range(num_step)]
            self.layers.append(nn.ModuleList(layer))
            prior = NICE(in_channels, hidden_channels=hidden_channels, s_channels=s_channels, scale=True, inverse=inverse, factor=factor)
            self.priors.append(prior)
            in_channels = in_channels - channel_step
            assert in_channels == prior.z1_channels
            factor = factor - 1
        self.z1_channels = in_channels

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        # [batch]
        logdet_accum = input.new_zeros(input.size(0))
        outputs = []
        for layer, prior in zip(self.layers, self.priors):
            for step in layer:
                out, logdet = step.forward(out, s=s)
                logdet_accum = logdet_accum + logdet
            out, logdet = prior.forward(out, s=s)
            logdet_accum = logdet_accum + logdet
            # split
            out1, out2 = split2d(out, prior.z1_channels)
            outputs.append(out2)
            out = out1

        outputs.append(out)
        outputs.reverse()
        out = unsplit2d(outputs)
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out = input
        outputs = []
        for prior in self.priors:
            out1, out2 = split2d(out, prior.z1_channels)
            outputs.append(out2)
            out = out1

        # [batch]
        logdet_accum = out.new_zeros(out.size(0))
        for layer, prior in zip(reversed(self.layers), reversed(self.priors)):
            out2 = outputs.pop()
            out = unsplit2d([out, out2])
            out, logdet = prior.backward(out, s=s)
            logdet_accum = logdet_accum + logdet
            for step in reversed(layer):
                out, logdet = step.backward(out, s=s)
                logdet_accum = logdet_accum + logdet

        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        out = data
        # [batch]
        logdet_accum = data.new_zeros(data.size(0))
        outputs = []
        for layer, prior in zip(self.layers, self.priors):
            for step in layer:
                out, logdet = step.init(out, s=s, init_scale=init_scale)
                logdet_accum = logdet_accum + logdet
            out, logdet = prior.init(out, s=s, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            # split
            out1, out2 = split2d(out, prior.z1_channels)
            outputs.append(out2)
            out = out1

        outputs.append(out)
        outputs.reverse()
        out = unsplit2d(outputs)
        return out, logdet_accum


class MaCow(Flow):
    """
    Masked Convolutional Flow
    """
    def __init__(self, levels, num_steps, in_channels, kernel_size, factors, hidden_channels=256, s_channels=0, dilations=None, scale=True, inverse=False, bottom=True):
        super(MaCow, self).__init__(inverse)
        assert levels > 1, 'MaCow should have at least 2 levels.'
        assert levels == len(num_steps)
        factors = [0] + factors + [0] if bottom else factors + [0]
        assert levels == len(factors)
        if dilations is None:
            dilations = [1] * levels
        blocks = []
        self.levels = levels
        self.internals = levels - 2 if bottom else levels - 1
        for level in range(levels):
            dilation = dilations[level]
            if level == 0 and bottom:
                macow_block = MaCowBottomBlock(num_steps[level], in_channels, kernel_size, s_channels, dilation, scale=scale, inverse=inverse)
                blocks.append(macow_block)
            elif level == levels - 1:
                in_channels = in_channels * 4
                s_channels = s_channels * 4
                macow_block = MaCowTopBlock(num_steps[level], in_channels, kernel_size, s_channels, dilation, scale=scale, inverse=inverse)
                blocks.append(macow_block)
            else:
                in_channels = in_channels * 4
                s_channels = s_channels * 4
                macow_block = MaCowInternalBlock(num_steps[level], in_channels, kernel_size, hidden_channels, s_channels, dilation,
                                                 factor=factors[level], scale=scale, inverse=inverse)
                blocks.append(macow_block)
                in_channels = macow_block.z1_channels
        self.blocks = nn.ModuleList(blocks)

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = input.new_zeros(input.size(0))
        out = input
        outputs = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, MaCowInternalBlock) or isinstance(block, MaCowTopBlock):
                if s is not None:
                    s = squeeze2d(s, factor=2)
                out = squeeze2d(out, factor=2)
            out, logdet = block.forward(out, s=s)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock):
                out1, out2 = split2d(out, block.z1_channels)
                outputs.append(out2)
                out = out1

        out = unsqueeze2d(out, factor=2)
        for _ in range(self.internals):
            out2 = outputs.pop()
            out = unsqueeze2d(unsplit2d([out, out2]), factor=2)
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = []
        if s is not None:
            s = squeeze2d(s, factor=2)
        out = squeeze2d(input, factor=2)
        for block in self.blocks:
            if isinstance(block, MaCowInternalBlock):
                if s is not None:
                    s = squeeze2d(s, factor=2)
                out1, out2 = split2d(out, block.z1_channels)
                outputs.append(out2)
                out = squeeze2d(out1, factor=2)

        logdet_accum = input.new_zeros(input.size(0))
        for i, block in enumerate(reversed(self.blocks)):
            if isinstance(block, MaCowInternalBlock):
                out2 = outputs.pop()
                out = unsplit2d([out, out2])
            out, logdet = block.backward(out, s=s)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock) or isinstance(block, MaCowTopBlock):
                if s is not None:
                    s = unsqueeze2d(s, factor=2)
                out = unsqueeze2d(out, factor=2)
        assert len(outputs) == 0
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        logdet_accum = data.new_zeros(data.size(0))
        out = data
        outputs = []
        for i, block in enumerate(self.blocks):
            if isinstance(block, MaCowInternalBlock) or isinstance(block, MaCowTopBlock):
                if s is not None:
                    s = squeeze2d(s, factor=2)
                out = squeeze2d(out, factor=2)
            out, logdet = block.init(out, s=s, init_scale=init_scale)
            logdet_accum = logdet_accum + logdet
            if isinstance(block, MaCowInternalBlock):
                out1, out2 = split2d(out, block.z1_channels)
                outputs.append(out2)
                out = out1

        out = unsqueeze2d(out, factor=2)
        for _ in range(self.internals):
            out2 = outputs.pop()
            out = unsqueeze2d(unsplit2d([out, out2]), factor=2)
        assert len(outputs) == 0
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "MaCow":
        return MaCow(**params)


MaCow.register('macow')
