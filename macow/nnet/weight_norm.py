__author__ = 'max'

from overrides import overrides
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn import Parameter

from macow.utils import norm


class NIN2d(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(NIN2d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = Parameter(torch.Tensor(out_features, in_features))
        self.weight_g = Parameter(torch.Tensor(out_features, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_v, mean=0.0, std=0.05)
        self.weight_g.data.copy_(norm(self.weight_v, 0))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def compute_weight(self):
        return self.weight_v * (self.weight_g / norm(self.weight_v, 0))

    def forward(self, input):
        weight = self.compute_weight()
        out = torch.einsum('...cij,oc->...oij', (input, weight))
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            out = self(x)
            out_features, height, width = out.size()[-3:]
            assert out_features == self.out_features
            # [batch, out_features, h * w] - > [batch, h * w, out_features]
            out = out.view(-1, out_features, height * width).transpose(1, 2)
            # [batch*height*width, out_features]
            out = out.contiguous().view(-1, out_features)
            # [out_features]
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.bias is not None:
                mean = mean.view(out_features, 1, 1)
                inv_stdv = inv_stdv.view(out_features, 1, 1)
                self.bias.add_(-mean).mul_(inv_stdv)
            return self(x)


class NIN4d(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(NIN4d, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_v = Parameter(torch.Tensor(out_features, in_features))
        self.weight_g = Parameter(torch.Tensor(out_features, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1, 1, 1, 1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_v, mean=0.0, std=0.05)
        self.weight_g.data.copy_(norm(self.weight_v, 0))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def compute_weight(self):
        return self.weight_v * (self.weight_g / norm(self.weight_v, 0))

    def forward(self, input):
        weight = self.compute_weight()
        out = torch.einsum('bc...,oc->bo...', (input, weight))
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            out = self(x)
            batch, out_features = out.size()[:2]
            assert out_features == self.out_features
            # [batch, out_features, h * w] - > [batch, h * w, out_features]
            out = out.view(batch, out_features, -1).transpose(1, 2)
            # [batch*height*width, out_features]
            out = out.contiguous().view(-1, out_features)
            # [out_features]
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.bias is not None:
                mean = mean.view(out_features, 1, 1, 1, 1)
                inv_stdv = inv_stdv.view(out_features, 1, 1, 1, 1)
                self.bias.add_(-mean).mul_(inv_stdv)
            return self(x)


class LinearWeightNorm(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearWeightNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.05)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)
        self.linear = nn.utils.weight_norm(self.linear)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # [batch, out_features]
            out = self(x).view(-1, self.linear.out_features)
            # [out_features]
            mean = out.mean(dim=0)
            std = out.std(dim=0)
            inv_stdv = init_scale / (std + 1e-6)

            self.linear.weight_g.mul_(inv_stdv.unsqueeze(1))
            if self.linear.bias is not None:
                self.linear.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.linear(input)


class Conv2dWeightNorm(nn.Module):
    """
    Conv2d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dWeightNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.conv.weight, mean=0.0, std=0.05)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
        self.conv = nn.utils.weight_norm(self.conv)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out = self(x)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            # [n_channels]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-6)

            self.conv.weight_g.mul_(inv_stdv.view(n_channels, 1, 1, 1))
            if self.conv.bias is not None:
                self.conv.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.conv(input)

    @overrides
    def extra_repr(self):
        return self.conv.extra_repr()


class ConvTranspose2dWeightNorm(nn.Module):
    """
    Convolution transpose 2d with weight normalization
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super(ConvTranspose2dWeightNorm, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                         padding=padding, output_padding=output_padding, groups=groups,
                                         bias=bias, dilation=dilation)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.deconv.weight, mean=0.0, std=0.05)
        if self.deconv.bias is not None:
            nn.init.constant_(self.deconv.bias, 0)
        self.deconv = nn.utils.weight_norm(self.deconv, dim=1)

    def _output_padding(self, input, output_size):
        return self.deconv._output_padding(input, output_size)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out = self(x)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            # [n_channels]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-6)

            self.deconv.weight_g.mul_(inv_stdv.view(1, n_channels, 1, 1))
            if self.deconv.bias is not None:
                self.deconv.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        return self.deconv(input)

    @overrides
    def extra_repr(self):
        return self.deconv.extra_repr()


class MaskedConv2d(nn.Module):
    """
    Conv2d with mask and weight normalization.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 mask_type='A', order='A', masked_channels=None,
                 stride=1, dilation=1, groups=1):
        super(MaskedConv2d, self).__init__()
        assert mask_type in {'A', 'B'}
        assert order in {'A', 'B'}
        self.mask_type = mask_type
        self.order = order
        kernel_size = _pair(kernel_size)
        for k in kernel_size:
            assert k % 2 == 1, 'kernel cannot include even number: {}'.format(self.kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        stride = _pair(stride)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        # masked all input channels by default
        masked_channels = in_channels if masked_channels is None else masked_channels
        self.masked_channels = masked_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight_v = Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        self.weight_g = Parameter(torch.Tensor(out_channels, 1, 1, 1))
        self.bias = Parameter(torch.Tensor(out_channels))

        self.register_buffer('mask', torch.ones(self.weight_v.size()))
        _, _, kH, kW = self.weight_v.size()
        mask = np.ones([*self.mask.size()], dtype=np.float32)
        mask[:, :masked_channels, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        mask[:, :masked_channels, kH // 2 + 1:] = 0

        # reverse order
        if order == 'B':
            reverse_mask = mask[:, :, ::-1, :]
            reverse_mask = reverse_mask[:, :, :, ::-1]
            mask = reverse_mask.copy()
        self.mask.copy_(torch.from_numpy(mask).float())
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_v, mean=0.0, std=0.05)
        self.weight_v.data.mul_(self.mask)
        _norm = norm(self.weight_v, 0).data + 1e-8
        self.weight_g.data.copy_(_norm.log())
        nn.init.constant_(self.bias, 0)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            # [batch, n_channels, H, W]
            out = self(x)
            n_channels = out.size(1)
            out = out.transpose(0, 1).contiguous().view(n_channels, -1)
            # [n_channels]
            mean = out.mean(dim=1)
            std = out.std(dim=1)
            inv_stdv = init_scale / (std + 1e-6)
            self.weight_g.add_(inv_stdv.log().view(n_channels, 1, 1, 1))
            self.bias.add_(-mean).mul_(inv_stdv)
            return self(x)

    def forward(self, input):
        self.weight_v.data.mul_(self.mask)
        _norm = norm(self.weight_v, 0) + 1e-8
        weight = self.weight_v * (self.weight_g.exp() / _norm)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}({masked_channels}), {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', type={mask_type}, order={order}'
        return s.format(**self.__dict__)


class ShiftedConv2d(Conv2dWeightNorm):
    """
    Conv2d with shift operation.
    A -> top
    B -> bottom
    C -> left
    D -> right
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), dilation=1, groups=1, bias=True, order='A'):
        assert len(stride) == 2
        assert len(kernel_size) == 2
        assert order in {'A', 'B', 'C', 'D'}, 'unknown order: {}'.format(order)
        if order in {'A', 'B'}:
            assert kernel_size[1] % 2 == 1, 'kernel width cannot be even number: {}'.format(kernel_size)
        else:
            assert kernel_size[0] % 2 == 1, 'kernel height cannot be even number: {}'.format(kernel_size)

        self.order = order
        if order == 'A':
            # left, right, top, bottom
            self.shift_padding = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2, kernel_size[0], 0)
            # top, bottom, left, right
            self.cut = (0, -1, 0, 0)
        elif order == 'B':
            # left, right, top, bottom
            self.shift_padding = ((kernel_size[1] - 1) // 2, (kernel_size[1] - 1) // 2, 0, kernel_size[0])
            # top, bottom, left, right
            self.cut = (1, 0, 0, 0)
        elif order == 'C':
            # left, right, top, bottom
            self.shift_padding = (kernel_size[1], 0, (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
            # top, bottom, left, right
            self.cut = (0, 0, 0, -1)
        elif order == 'D':
            # left, right, top, bottom
            self.shift_padding = (0, kernel_size[1], (kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)
            # top, bottom, left, right
            self.cut = (0, 0, 1, 0)
        else:
            raise ValueError('unknown order: {}'.format(order))

        super(ShiftedConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=0,
                                            stride=stride, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input, shifted=True):
        if shifted:
            input = F.pad(input, self.shift_padding)
            bs, channels, height, width = input.size()
            t, b, l, r = self.cut
            input = input[:, :, t:height + b, l:width + r]
        return self.conv(input)

    @overrides
    def extra_repr(self):
        s = self.conv.extra_repr()
        s += ', order={order}'
        s += ', shift_padding={shift_padding}'
        s += ', cut={cut}'
        return s.format(**self.__dict__)
