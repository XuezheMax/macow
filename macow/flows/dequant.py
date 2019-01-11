__author__ = 'max'

from overrides import overrides
from collections import OrderedDict
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.activation import SigmoidFlow
from macow.flows.macow import MaCow
from macow.nnet.resnet import ResNet
from macow.nnet.weight_norm import Conv2dWeightNorm, ConvTranspose2dWeightNorm


class DeQuantFlow(Flow):
    def __init__(self, levels, num_steps, in_channels, kernel_size, factors, hidden_channels=512, s_channels=0, scale=True, dropout=0.0):
        super(DeQuantFlow, self).__init__(False)
        self.macow = MaCow(levels, num_steps, in_channels, kernel_size, factors,
                           hidden_channels=hidden_channels, s_channels=s_channels, scale=scale, dropout=dropout, inverse=False, bottom=True)
        self.sigmoid = SigmoidFlow(inverse=False)
        if s_channels > 0:
            layers = list()
            layers.append(('resnet0', ResNet(in_channels, [32, 32], [1, 1])))
            planes = 32
            out_planes = [planes]
            for level in range(1, levels):
                layers.append(('down%d' % level, Conv2dWeightNorm(planes, planes * 2, 3, 2, 1, bias=True)))
                layers.append(('elu%d' % level, nn.ELU(inplace=True)))
                planes = min(planes * 2, 96)
                out_planes.append(planes)
                layers.append(('resnet%d' % level, ResNet(planes, [planes, planes], [1, 1])))

            planes = out_planes.pop()
            for level in range(1, levels):
                layers.append(('up%d' % level, ConvTranspose2dWeightNorm(planes, out_planes[-1], 3, 2, 1, 1, bias=True)))
                layers.append(('elu%d' % (level + levels - 1), nn.ELU(inplace=True)))
                planes = out_planes.pop()

            layers.append(('s_level', Conv2dWeightNorm(planes, s_channels, 1, bias=True)))
            assert len(out_planes) == 0
            self.encoder = nn.Sequential(OrderedDict(layers))
        else:
            self.encoder = None

    def init_encoder(self, s, init_scale=1.0) -> torch.Tensor:
        out = s
        for layer in self.encoder:
            if isinstance(layer, nn.ELU):
                out = layer(out)
            else:
                out = layer.init(out, init_scale=init_scale)
        return out

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is not None:
            s = self.encoder(s)
        out, logdet_accum = self.macow.forward(input, s=s)
        out, logdet = self.sigmoid.forward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is not None:
            s = self.encoder(s)
        out, logdet_accum = self.sigmoid.backward(input)
        out, logdet = self.macow.backward(out, s=s)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is not None:
            s = self.init_encoder(s, init_scale=init_scale)
        out, logdet_accum = self.macow.init(data, s=s, init_scale=init_scale)
        out, logdet = self.sigmoid.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "DeQuantFlow":
        return DeQuantFlow(**params)
