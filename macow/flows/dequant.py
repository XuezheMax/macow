__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch

from macow.flows.flow import Flow
from macow.flows.activation import SigmoidFlow
from macow.flows.macow import MaCow


class DeQuantFlow(Flow):
    def __init__(self, levels, num_steps, in_channels, kernel_size, factors, hidden_channels=512, s_channels=0, scale=True, dropout=0.0):
        super(DeQuantFlow, self).__init__(False)
        self.macow = MaCow(levels, num_steps, in_channels, kernel_size, factors,
                           hidden_channels=hidden_channels, s_channels=s_channels, scale=scale, dropout=dropout, inverse=False, bottom=True)
        self.sigmoid = SigmoidFlow(inverse=False)

    @overrides
    def forward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.macow.forward(input, s=s)
        out, logdet = self.sigmoid.forward(out)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def backward(self, input: torch.Tensor, s=None) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.sigmoid.backward(input)
        out, logdet = self.macow.backward(out, s=s)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @overrides
    def init(self, data, s=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        out, logdet_accum = self.macow.init(data, s=s, init_scale=init_scale)
        out, logdet = self.sigmoid.init(out, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return out, logdet_accum

    @classmethod
    def from_params(cls, params: Dict) -> "DeQuantFlow":
        return DeQuantFlow(**params)
