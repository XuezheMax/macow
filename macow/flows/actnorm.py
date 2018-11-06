__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter

from macow.flows.flow import Flow


class ActNormFlow(Flow):
    def __init__(self, in_features, inverse=False):
        super(ActNormFlow, self).__init__(inverse)
        self.in_features = in_features
        self.log_scale = Parameter(torch.Tensor(in_features))
        self.bias = Parameter(torch.Tensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, N1, N2, ..., in_channels]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_channels], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        out = input * self.log_scale.exp() + self.bias
        logdet = self.log_scale.sum(dim=0, keepdim=True)
        if input.dim() > 2:
            num = np.prod(input.size()[1:-1])
            logdet = logdet * num.astype(float)
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: input: Tensor
                input tensor [batch, N1, N2, ..., in_channels]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_channels], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        out = input - self.bias
        out = out.div(self.log_scale.exp() + 1e-8)
        logdet = self.log_scale.sum(dim=0, keepdim=True) * -1.0
        if input.dim() > 2:
            num = np.prod(input.size()[1:-1])
            logdet = logdet * num.astype(float)
        return out, logdet

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            data: input: Tensor
                input tensor [batch, N1, N2, ..., in_channels]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, N1, N2, ..., in_channels], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        with torch.no_grad():
            out, _ = self.forward(data, h=h)
            mean = out.view(-1, self.in_features).mean(dim=0)
            std = out.view(-1, self.in_features).std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.log_scale.add_(inv_stdv.log())
            self.bias.add_(-mean).mul_(inv_stdv)
            return self.forward(data, h=h)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_features={}'.format(self.inverse, self.in_features)

    @classmethod
    def from_params(cls, params: Dict) -> "ActNormFlow":
        return ActNormFlow(**params)


class ActNorm2dFlow(Flow):
    def __init__(self, in_channels, inverse=False):
        super(ActNorm2dFlow, self).__init__(inverse)
        self.in_channels = in_channels
        self.log_scale = Parameter(torch.Tensor(in_channels, 1, 1))
        self.bias = Parameter(torch.Tensor(in_channels, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.log_scale, mean=0, std=0.05)
        nn.init.constant_(self.bias, 0.)

    @overrides
    def forward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        batch, channels, H, W = input.size()
        out = input * self.log_scale.exp() + self.bias
        logdet = self.log_scale.sum(dim=0).squeeze(1) * H * W
        return out, logdet

    @overrides
    def backward(self, input: torch.Tensor, h=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            input: Tensor
                input tensor [batch, in_channels, H, W]
            h: Tensor
                conditional input (default: None)

        Returns: out: Tensor , logdet: Tensor
            out: [batch, in_channels, H, W], the output of the flow
            logdet: [batch], the log determinant of :math:`\partial output / \partial input`

        """
        batch, channels, H, W = input.size()
        out = input - self.bias
        out = out.div(self.log_scale.exp() + 1e-8)
        logdet = self.log_scale.sum(dim=0).squeeze(1) * H * W * -1.0
        return out, logdet

    @overrides
    def init(self, data, h=None, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out, _ = self.forward(data, h=h)
            out = out.transpose(0, 1).contiguous().view(self.in_channels, -1)
            # [n_channels, 1, 1]
            mean = out.mean(dim=1).view(self.in_channels, 1, 1)
            std = out.std(dim=1).view(self.in_channels, 1, 1)
            inv_stdv = init_scale / (std + 1e-6)

            self.log_scale.add_(inv_stdv.log())
            self.bias.add_(-mean).mul_(inv_stdv)
            return self.forward(data, h=h)

    @overrides
    def extra_repr(self):
        return 'inverse={}, in_channels={}'.format(self.inverse, self.in_channels)

    @classmethod
    def from_params(cls, params: Dict) -> "ActNorm2dFlow":
        return ActNorm2dFlow(**params)


ActNormFlow.register('actnorm')
ActNorm2dFlow.register('actnorm2d')
