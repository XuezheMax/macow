__author__ = 'max'

import os
import json
import numbers
import math
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.parallel import DataParallelFlow
from macow.flows.depth_scale import DepthScaleBlock


class FlowGenModel(nn.Module):
    """
    Flow-based Generative model
    """
    def __init__(self, flow: Flow, ngpu=1):
        super(FlowGenModel, self).__init__()
        assert flow.inverse, 'flow based generative should have inverse mode'
        self.flow = flow
        assert ngpu > 0, 'the number of GPUs should be positive.'
        self.ngpu = ngpu
        if ngpu > 1:
            self.flow = DataParallelFlow(self.flow, device_ids=list(range(ngpu)))

        self.beta = torch.distributions.beta.Beta(2, 2)

    def dequantize(self, x, nsamples=1) -> Tuple[torch.Tensor, torch.Tensor]:
        # [batch, nsamples, channels, H, W]
        # return x.new_empty(x.size(0), nsamples, *x.size()[1:]).uniform_(), x.new_zeros(x.size(0), nsamples)
        noise = self.beta.rsample((x.size(0), nsamples, *x.size()[1:]))
        log_posterior = self.beta.log_prob(noise)
        log_posterior = log_posterior.view(x.size(0), nsamples, -1).sum(dim=2)
        return noise.type_as(x), log_posterior.type_as(x)

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor
                The input data with shape =[batch, x_shape]

        Returns: z: Tensor, logdet: Tensor, eps: List[Tensor]
            z, the latent variable
            logdet, the log determinant of :math:`\partial z / \partial x`
            Then the density :math:`\log(p(x)) = \log(p(z)) + logdet`
            eps: eps for multi-scale architecture.
        """
        z, logdet = self.flow.bwdpass(x)
        return z, logdet

    def decode(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            z: Tensor
                The latent code with shape =[batch, *]

        Returns: x: Tensor, logdet: Tensor
            x, the decoded variable
            logdet, the log determinant of :math:`\partial z / \partial x`
            Then the density :math:`\log(p(x)) = \log(p(z)) + logdet`
        """
        x, logdet = self.flow.fwdpass(z)
        return x, logdet

    def init(self, data, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.bwdpass(data, init=True, init_scale=init_scale)

    def log_probability(self, x) -> torch.Tensor:
        """

        Args:
            x: Tensor
                The input data with shape =[batch, x_shape]

        Returns:
            Tensor
            The tensor of the posterior probabilities of x shape = [batch]
        """
        # [batch, x_shape]
        z, logdet = self.encode(x)
        # [batch, x_shape] --> [batch, numels]
        z = z.view(z.size(0), -1)
        # [batch]
        log_probs = z.mul(z).sum(dim=1) + math.log(math.pi * 2.)* z.size(1)
        return log_probs.mul(-0.5) + logdet

    @classmethod
    def from_params(cls, params: Dict) -> "FlowGenModel":
        flow_params = params.pop('flow')
        flow = Flow.by_name(flow_params.pop('type')).from_params(flow_params)
        return FlowGenModel(flow, **params)

    @classmethod
    def load(cls, model_path, device) -> "FlowGenModel":
        params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        model_name = os.path.join(model_path, 'model.pt')
        fgen = FlowGenModel.from_params(params)
        fgen.load_state_dict(torch.load(model_name, map_location=device))
        return fgen.to(device)


class DepthScaleFlowModel(nn.Module):
    """
    Flow-based Generative model
    """
    def __init__(self, fundation: Flow, bottom1: Flow, bottom2: Flow, bottom3: Flow, ngpu=1):
        super(DepthScaleFlowModel, self).__init__()
        assert fundation.inverse and bottom1.inverse and bottom2.inverse and bottom3, 'flow based generative should have inverse mode'
        self.fundation = fundation
        self.bottom1 = bottom1
        self.bottom2 = bottom2
        self.bottom3 = bottom3
        assert ngpu > 0, 'the number of GPUs should be positive.'
        self.ngpu = ngpu
        if ngpu > 1:
            self.fundation = DataParallelFlow(self.fundation, device_ids=list(range(ngpu)))
            self.bottom1 = DataParallelFlow(self.bottom1, device_ids=list(range(ngpu)))
            self.bottom2 = DataParallelFlow(self.bottom2, device_ids=list(range(ngpu)))
            self.bottom3 = DataParallelFlow(self.bottom3, device_ids=list(range(ngpu)))
        self.bottoms = nn.ModuleList([self.bottom1, self.bottom2, self.bottom3])

    def depth_8to5bits(self, img8bits):
        x_5bits, logdet = self.bottom1.bwdpass(img8bits)
        return x_5bits, logdet

    def depth_5to3bits(self, img5bits):
        x_3bits, logdet = self.bottom2.bwdpass(img5bits)
        return x_3bits, logdet

    def depth_downscale_3bits(self, img3bits):
        x, logdet_accum = self.bottom3.bwdpass(img3bits)
        return x, logdet_accum

    def depth_downscale_5bits(self, img5bits):
        x_3bits, logdet_accum = self.depth_5to3bits(img5bits)
        x, logdet = self.depth_downscale_3bits(x_3bits)
        logdet_accum = logdet_accum + logdet
        return x, logdet_accum, x_3bits

    def depth_downscale_8bits(self, img8bits):
        x_5bits, logdet_accum = self.depth_8to5bits(img8bits)
        x_3bits, logdet = self.depth_5to3bits(x_5bits)
        logdet_accum = logdet_accum + logdet
        x, logdet = self.depth_downscale_3bits(x_3bits)
        logdet_accum = logdet_accum + logdet
        return x, logdet_accum, x_5bits

    def depth_upscale_3bits(self, x):
        x_3bits, logdet_accum = self.bottom3.fwdpass(x)
        return x_3bits, logdet_accum

    def depth_upscale_5bits(self, x):
        x_3bits, logdet_accum = self.depth_upscale_3bits(x)
        x_5bits, logdet = self.bottom2.fwdpass(x_3bits)
        logdet_accum = logdet_accum + logdet
        return x_5bits, logdet_accum

    def depth_upscale_8bits(self, x):
        x_5bits, logdet_accum = self.depth_upscale_5bits(x)
        x_8bits, logdet = self.bottom1.fwdpass(x_5bits)
        logdet_accum = logdet_accum + logdet
        return x_8bits, logdet_accum

    def depth_upscale(self, x):
        x_3bits, logdet_accum = self.bottom3.fwdpass(x)
        x_5bits, logdet = self.bottom2.fwdpass(x_3bits)
        logdet_accum = logdet_accum + logdet
        x_8bits, logdet = self.bottom1.fwdpass(x_5bits)
        logdet_accum = logdet_accum + logdet
        return x_8bits, x_5bits, x_3bits, logdet_accum

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            x: Tensor
                The input data with shape =[batch, x_shape]

        Returns: z: Tensor, logdet: Tensor, eps: List[Tensor]
            z, the latent variable
            logdet, the log determinant of :math:`\partial z / \partial x`
            Then the density :math:`\log(p(x)) = \log(p(z)) + logdet`
            eps: eps for multi-scale architecture.
        """
        z, logdet = self.fundation.bwdpass(x)
        return z, logdet

    def decode(self, z) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        Args:
            z: Tensor
                The latent code with shape =[batch, *]

        Returns: x: Tensor, logdet: Tensor
            x, the decoded variable
            logdet, the log determinant of :math:`\partial z / \partial x`
            Then the density :math:`\log(p(x)) = \log(p(z)) + logdet`
        """
        x, logdet = self.fundation.fwdpass(z)
        return x, logdet

    def init(self, img8bits, img5bits, img3bits, init_scale=1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        x_5bits, logdet_accum = self.bottom1.bwdpass(img8bits, init=True, init_scale=init_scale)
        x_3bits, logdet = self.bottom2.bwdpass(img5bits, init=True, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        x, logdet = self.bottom3.bwdpass(img3bits, init=True, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        z, logdet = self.fundation.bwdpass(x, init=True, init_scale=init_scale)
        logdet_accum = logdet_accum + logdet
        return z, logdet_accum

    def log_probability(self, x, logdet_bottom) -> torch.Tensor:
        """

        Args:
            x: Tensor
                The input data with shape =[batch, x_shape]

        Returns:
            Tensor
            The tensor of the posterior probabilities of x shape = [batch]
        """
        # [batch, x_shape]
        z, logdet = self.encode(x)
        # [batch, x_shape] --> [batch, numels]
        z = z.view(z.size(0), -1)
        # [batch]
        log_probs = z.mul(z).sum(dim=1) + math.log(math.pi * 2.)* z.size(1)
        return log_probs.mul(-0.5) + logdet + logdet_bottom

    @classmethod
    def from_params(cls, params: Dict) -> "DepthScaleFlowModel":
        fundation_params = params.pop('fundation')
        fundation = Flow.by_name(fundation_params.pop('type')).from_params(fundation_params)
        bottom_params = params.pop('bottom')
        bottom1 = DepthScaleBlock.from_params(bottom_params)
        bottom2 = DepthScaleBlock.from_params(bottom_params)
        bottom3 = DepthScaleBlock.from_params(bottom_params)
        return DepthScaleFlowModel(fundation, bottom1, bottom2, bottom3, **params)

    @classmethod
    def load(cls, model_path, device) -> "DepthScaleFlowModel":
        params = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
        model_name = os.path.join(model_path, 'model.pt')
        dsgen = DepthScaleFlowModel.from_params(params)
        dsgen.load_state_dict(torch.load(model_name, map_location=device))
        return dsgen.to(device)
