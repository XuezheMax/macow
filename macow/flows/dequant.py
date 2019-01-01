__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import torch
import torch.nn as nn

from macow.flows.flow import Flow
from macow.flows.macow import MaCow


class DeQuantFlow(Flow):
    def __init__(self, levels, num_steps, in_channels, kernel_size, factors, hidden_channels=512, s_channels=0, scale=True, dropout=0.0):
        super(DeQuantFlow, self).__init__(False)
        self.macow = MaCow(levels, num_steps, in_channels, kernel_size, factors,
                           hidden_channels=hidden_channels, s_channels=s_channels, scale=scale, dropout=dropout, inverse=False, bottom=True)
