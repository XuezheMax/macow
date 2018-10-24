__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import math
import torch
import torch.nn.functional as F

from macow.flows.flow import Flow


class LeakyReLUFlow(Flow):
    def __init__(self, negative_slope=0.1, inverse=False):
        super(LeakyReLUFlow, self).__init__(inverse)
        self.negative_slope = negative_slope

    @overrides
    def forward(self, input: torch.Tensor, *h) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]
            *h:

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        out = F.leaky_relu(input, self.negative_slope, False)
        log_slope = math.log(self.negative_slope)
        # [batch]
        logdet = input.view(input.size(0), -1).lt(0.0).type_as(input).sum(dim=1) * log_slope
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, *h) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]
            *h:

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        negative_slope = 1.0 / self.negative_slope
        out = F.leaky_relu(input, negative_slope, False)
        log_slope = math.log(negative_slope)
        # [batch]
        logdet = input.view(input.size(0), -1).lt(0.0).type_as(input).sum(dim=1) * log_slope
        return out, logdet

    @overrides
    def init(self, data, *h, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, negative_slope={}'.format(self.inverse, self.negative_slope)

    @classmethod
    def from_params(cls, params: Dict) -> "LeakyReLUFlow":
        return LeakyReLUFlow(**params)


class ELUFlow(Flow):
    def __init__(self, alpha=1.0, inverse=False):
        super(ELUFlow, self).__init__(inverse)
        self.alpha = alpha

    @overrides
    def forward(self, input: torch.Tensor, *h) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]
            *h:

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        out = F.elu(input, self.alpha, False)
        # [batch, numel]
        input = input.view(input.size(0), -1)
        logdet = input + math.log(self.alpha)
        # [batch]
        logdet = (input.lt(0.0).float() * logdet).sum(dim=1)
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, *h) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, *]
            *h:

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        mask = input.lt(0.0).float()
        out = input * (1.0 - mask) + mask * (input.div(self.alpha) + 1 + 1e-10).log()
        # [batch, numel]
        out_flat = out.view(input.size(0), -1)
        logdet = out_flat + math.log(self.alpha)
        # [batch]
        logdet = (mask.view(out_flat.size()) * logdet).sum(dim=1).mul(-1.0)
        return out, logdet

    @overrides
    def init(self, data, *h, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            return self.forward(data)

    @overrides
    def extra_repr(self):
        return 'inverse={}, alpha={}'.format(self.inverse, self.alpha)

    @classmethod
    def from_params(cls, params: Dict) -> "ELUFlow":
        return ELUFlow(**params)


LeakyReLUFlow.register('leaky_relu')
ELUFlow.register('elu')
