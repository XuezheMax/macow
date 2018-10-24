__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from macow.flows.flow import Flow
from macow.flows.parallel import DataParallelFlow


class FGenModel(nn.Module):
    """
    Flow-based Generative model
    """
    def __init__(self, flow: Flow, ngpu=1):
        super(FGenModel, self).__init__()
        self.flow = flow
        assert ngpu > 0, 'the number of GPUs should be positive.'
        self.ngpu = ngpu
        if ngpu > 1:
            self.flow = DataParallelFlow(self.flow)
