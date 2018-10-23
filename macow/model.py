__author__ = 'max'

from overrides import overrides
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from macow.flows.flow import Flow


class FGenModel(nn.Module):
    """
    Flow-based Generative model
    """