__author__ = 'max'

from overrides import overrides
import math
import torch
import torch.nn as nn
from macow.nnet.weight_norm import LinearWeightNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, features, heads):
        super(MultiHeadAttention, self).__init__()
        self.proj = LinearWeightNorm(features, 3 * features, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        assert features % heads == 0
        self.features = features
        self.heads = heads

    @overrides
    def forward(self, x, pos_enc=None):
        if pos_enc is not None:
            x = x + pos_enc
        bs, timesteps, features = x.size()
        heads = self.heads
        dim = features // heads
        # [batch, timesteps, 3 * features]
        c = self.proj(x)
        # [batch, timesteps, 3, heads, dim]
        c = c.view(bs, timesteps, 3, heads, dim)
        # [3, batch, heads, timesteps, dim]
        c = c.permute(2, 0, 3, 1, 4)
        # [batch, heads, timesteps, dim]
        queries = c[0]
        keys = c[1]
        values = c[2]
        # attention weights [batch, heads, timesteps, timesteps]
        attn_weights = torch.matmul(queries, keys.transpose(2, 3)).div(math.sqrt(dim))
        attn_weights = self.softmax(attn_weights)
        # values [batch, heads, timesteps, dim]
        out = torch.matmul(attn_weights, values)
        # merge heads
        # [batch, timesteps, heads, dim]
        out = x.view(bs, timesteps, heads, dim) + out.transpose(1, 2)
        out = out.view(bs, timesteps, features)
        return out

    def init(self, x, pos_enc=None, init_scale=1.0):
        if pos_enc is not None:
            x = x + pos_enc
        bs, timesteps, features = x.size()
        heads = self.heads
        dim = features // heads
        # [batch, timesteps, 3 * features]
        c = self.proj.init(x, init_scale=init_scale)
        # [batch, timesteps, 3, heads, dim]
        c = c.view(bs, timesteps, 3, heads, dim)
        # [3, batch, heads, timesteps, dim]
        c = c.permute(2, 0, 3, 1, 4)
        # [batch, heads, timesteps, dim]
        queries = c[0]
        keys = c[1]
        values = c[2]
        # attention weights [batch, heads, timesteps, timesteps]
        attn_weights = torch.matmul(queries, keys.transpose(2, 3)).div(math.sqrt(dim))
        attn_weights = self.softmax(attn_weights)
        # values [batch, heads, timesteps, dim]
        out = torch.matmul(attn_weights, values)
        # merge heads
        # [batch, timesteps, heads, dim]
        out = x.view(bs, timesteps, heads, dim) + out.transpose(1, 2)
        out = out.view(bs, timesteps, features)
        return out
