__author__ = 'max'

import torch


def norm(p: torch.Tensor, dim: int):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return norm(p.transpose(0, dim), 0).transpose(0, dim)


def squeeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    batch, n_channels, height, width = x.size()
    assert height % factor == 0 and width % factor == 0
    # [batch, channels, height, width] -> [batch, channels, height/factor, factor, width/factor, factor]
    x = x.view(-1, n_channels, height // factor, factor, width // factor, factor)
    # [batch, channels, factor, factor, height/factor, width/factor]
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    # [batch, channels*factor*factor, height/factor, width/factor]
    x = x.view(-1, n_channels * factor * factor, height // factor, width // factor)
    return x


def unsqueeze2d(x, factor=2):
    assert factor >= 1
    if factor == 1:
        return x
    batch, n_channels, height, width = x.size()
    assert n_channels >= 4 and n_channels % 4 == 0
    # [batch, channels, height, width] -> [batch, channels/(factor*factor), factor, factor, height, width]
    x = x.view(-1, int(n_channels / factor ** 2), factor, factor, height, width)
    # [batch, channels/(factor*factor), height, factor, width, factor]
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    # [batch, channels/(factor*factor), height*factor, width*factor]
    x = x.view(-1, int(n_channels / factor ** 2), int(height * factor), int(width * factor))
    return x
