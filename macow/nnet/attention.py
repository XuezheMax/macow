__author__ = 'max'

from overrides import overrides
import math
import torch
import torch.nn as nn
from macow.nnet.weight_norm import LinearWeightNorm, Conv2dWeightNorm
from macow.utils import gate


class MultiHeadAttention(nn.Module):
    def __init__(self, features, heads):
        super(MultiHeadAttention, self).__init__()
        self.proj1 = LinearWeightNorm(features, 3 * features, bias=True)
        self.proj2 = LinearWeightNorm(features, 2 * features, bias=True)
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
        c = self.proj1(x)
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
        # [batch, timesteps, heads, dim] -> [batch, timesteps, features]
        out = out.transpose(1, 2).contiguous().view(bs, timesteps, features)
        out1, out2 = self.proj2(out).chunk(2, dim=2)
        out = gate(out1, out2) + x
        return out

    def init(self, x, pos_enc=None, init_scale=1.0):
        if pos_enc is not None:
            x = x + pos_enc
        bs, timesteps, features = x.size()
        heads = self.heads
        dim = features // heads
        # [batch, timesteps, 3 * features]
        c = self.proj1.init(x, init_scale=init_scale)
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
        # [batch, timesteps, heads, dim] -> [batch, timesteps, features]
        out = out.transpose(1, 2).contiguous().view(bs, timesteps, features)
        out1, out2 = self.proj2.init(out, init_scale=0.1 * init_scale).chunk(2, dim=2)
        out = gate(out1, out2) + x
        return out


class MultiHeadAttention2d(nn.Module):
    def __init__(self, channels, heads):
        super(MultiHeadAttention2d, self).__init__()
        self.proj1 = Conv2dWeightNorm(channels, 3 * channels, 1, bias=True)
        self.proj2 = Conv2dWeightNorm(channels, 2 * channels, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        assert channels % heads == 0
        self.features = channels
        self.heads = heads

    @overrides
    def forward(self, x, pos_enc=None):
        # [batch, channels, height, width]
        if pos_enc is not None:
            x = x + pos_enc
        bs, channels, height, width = x.size()
        heads = self.heads
        dim = channels // heads
        # [batch, 3 * channels, height, width]
        c = self.proj1(x)
        # [batch, 3, heads, dim, height, width]
        c = c.view(bs, 3, heads, dim, height, width)
        # [batch, heads, dim, height, width]
        queries = c[:, 0]
        keys = c[:, 1]
        # [batch, heads, dim, timesteps]
        values = c[:, 2].view(bs, heads, dim, height * width)
        # attention weights [batch, heads, height, width, height, width]
        attn_weights = torch.einsum('bhdij,bhdkl->bhijkl', queries, keys).div(math.sqrt(dim))
        # attention weights [batch, heads, height, width, timesteps]
        attn_weights = self.softmax(attn_weights.view(bs, heads, height, width, -1))
        # values [batch, heads, dim, height, width]
        out = torch.einsum('bhdt,bhijt->bhdij', values, attn_weights)
        # merge heads
        # [batch, 2 * channels, heads, dim]
        out = self.proj2(out.view(bs, channels, height, width))
        out1, out2 = out.chunk(2, dim=1)
        out = gate(out1, out2) + x
        return out

    def init(self, x, pos_enc=None, init_scale=1.0):
        # [batch, channels, height, width]
        if pos_enc is not None:
            x = x + pos_enc
        bs, channels, height, width = x.size()
        heads = self.heads
        dim = channels // heads
        # [batch, 3 * channels, height, width]
        c = self.proj1.init(x, init_scale=init_scale)
        # [batch, 3, heads, dim, height, width]
        c = c.view(bs, 3, heads, dim, height, width)
        # [batch, heads, dim, height, width]
        queries = c[:, 0]
        keys = c[:, 1]
        # [batch, heads, dim, timesteps]
        values = c[:, 2].view(bs, heads, dim, height * width)
        # attention weights [batch, heads, height, width, height, width]
        attn_weights = torch.einsum('bhdij,bhdkl->bhijkl', queries, keys).div(math.sqrt(dim))
        # attention weights [batch, heads, height, width, timesteps]
        attn_weights = self.softmax(attn_weights.view(bs, heads, height, width, -1))
        # values [batch, heads, dim, height, width]
        out = torch.einsum('bhdt,bhijt->bhdij', values, attn_weights)
        # merge heads
        # [batch, 2 * channels, heads, dim]
        out = self.proj2.init(out.view(bs, channels, height, width), init_scale=0.1 * init_scale)
        out1, out2 = out.chunk(2, dim=1)
        out = gate(out1, out2) + x
        return out
