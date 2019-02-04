__author__ = 'max'

from overrides import overrides
import math
import torch
import torch.nn as nn
from macow.nnet.weight_norm import LinearWeightNorm, Conv2dWeightNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, features, heads, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.proj = LinearWeightNorm(features, 3 * features, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout, inplace=True)
        else:
            self.dropout = None
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
        if self.dropout is not None:
            out = self.dropout(out)
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
        if self.dropout is not None:
            out = self.dropout(out)
        # merge heads
        # [batch, timesteps, heads, dim]
        out = x.view(bs, timesteps, heads, dim) + out.transpose(1, 2)
        out = out.view(bs, timesteps, features)
        return out


class MultiHeadAttention2d(nn.Module):
    def __init__(self, channels, heads, dropout=0.0):
        super(MultiHeadAttention2d, self).__init__()
        self.proj = Conv2dWeightNorm(channels, 3 * channels, 1, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout, inplace=True)
        else:
            self.dropout = None
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
        c = self.proj(x)
        # [batch, 3, heads, dim, height, width]
        c = c.view(bs, 3, heads, dim, height, width)
        # [batch, heads, dim, height, width]
        queries = c[:, 0]
        keys = c[:, 1]
        # [batch, heads, dim, timesteps]
        values = c[:, 2].view(bs, heads, dim, height * width)
        # attention weights [batch, heads, height, width, height, width]
        attn_weights = torch.einsum('bhdij,bhdkl->bhijkl', (queries, keys)).div(math.sqrt(dim))
        # attention weights [batch, heads, height, width, timesteps]
        attn_weights = self.softmax(attn_weights.view(bs, heads, height, width, -1))
        # values [batch, heads, dim, height, width]
        out = torch.einsum('bhdt,bhijt->bhdij', (values, attn_weights))
        if self.dropout is not None:
            out = self.dropout(out)
        # merge heads
        # [batch, channels, heads, dim]
        out = x + out.view(bs, channels, height, width)
        return out

    def init(self, x, pos_enc=None, init_scale=1.0):
        # [batch, channels, height, width]
        if pos_enc is not None:
            x = x + pos_enc
        bs, channels, height, width = x.size()
        heads = self.heads
        dim = channels // heads
        # [batch, 3 * channels, height, width]
        c = self.proj.init(x, init_scale=init_scale)
        # [batch, 3, heads, dim, height, width]
        c = c.view(bs, 3, heads, dim, height, width)
        # [batch, heads, dim, height, width]
        queries = c[:, 0]
        keys = c[:, 1]
        # [batch, heads, dim, timesteps]
        values = c[:, 2].view(bs, heads, dim, height * width)
        # attention weights [batch, heads, height, width, height, width]
        attn_weights = torch.einsum('bhdij,bhdkl->bhijkl', (queries, keys)).div(math.sqrt(dim))
        # attention weights [batch, heads, height, width, timesteps]
        attn_weights = self.softmax(attn_weights.view(bs, heads, height, width, -1))
        # values [batch, heads, dim, height, width]
        out = torch.einsum('bhdt,bhijt->bhdij', (values, attn_weights))
        if self.dropout is not None:
            out = self.dropout(out)
        # merge heads
        # [batch, channels, heads, dim]
        out = x + out.view(bs, channels, height, width)
        return out
