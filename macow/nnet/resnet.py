__author__ = 'max'

import torch.nn as nn
from macow.nnet.weight_norm import Conv2dWeightNorm, ConvTranspose2dWeightNorm

__all__ = ['ResNet', 'DeResNet', ]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Conv2dWeightNorm(in_planes, out_planes,
                            kernel_size=3, stride=stride,
                            padding=1, bias=True)


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    "3x3 deconvolution with padding"
    return ConvTranspose2dWeightNorm(in_planes, out_planes,
                                     kernel_size=3, stride=stride, padding=1,
                                     output_padding=output_padding, bias=True)


class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.activation = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = Conv2dWeightNorm(inplanes, planes,
                                          kernel_size=1, stride=stride, bias=True)
        self.downsample = downsample
        self.stride = stride

    def init(self, x, init_scale=1.0):
        residual = x if self.downsample is None else self.downsample.init(x, init_scale=init_scale)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv1.init(x, init_scale=init_scale)
        out = self.activation(out)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv2.init(out, init_scale=init_scale)
        out = self.activation(out + residual)
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        return out

    def forward(self, x):
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        residual = x if self.downsample is None else self.downsample(x)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv1(x)
        out = self.activation(out)

        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        out = self.conv2(out)
        out = self.activation(out + residual)
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        return out


class DeResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, output_padding=0):
        super(DeResNetBlock, self).__init__()
        self.deconv1 = deconv3x3(inplanes, planes, stride, output_padding)
        self.activation = nn.ELU(inplace=True)
        self.deconv2 = deconv3x3(planes, planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = ConvTranspose2dWeightNorm(inplanes, planes,
                                                   kernel_size=1, stride=stride,
                                                   output_padding=output_padding, bias=True)
        self.downsample = downsample
        self.stride = stride

    def init(self, x, init_scale=1.0):
        x = self.activation(x)
        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        residual = x if self.downsample is None else self.downsample.init(x, init_scale=init_scale)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv1.init(x, init_scale=init_scale)
        out = self.activation(out)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv2.init(out, init_scale=init_scale)
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        return out + residual

    def forward(self, x):
        x = self.activation(x)
        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        residual = x if self.downsample is None else self.downsample(x)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv1(x)
        out = self.activation(out)

        # [batch, planes, h * stride - stride + 1 + op, w * stride - stride + 1 + op]
        out = self.deconv2(out)
        # [batch, planes, ceil(h/stride), ceil(w/stride)]
        return out + residual


class ResNet(nn.Module):
    def __init__(self, inplanes, planes, strides):
        super(ResNet, self).__init__()
        assert len(planes) == len(strides)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            block = ResNetBlock(inplanes, plane, stride=stride)
            blocks.append(block)
            inplanes = plane

        self.main = nn.Sequential(*blocks)

    def init(self, x, init_scale=1.0):
        for block in self.main:
            x = block.init(x, init_scale=init_scale)
        return x

    def forward(self, x):
        return self.main(x)


class DeResNet(nn.Module):
    def __init__(self, inplanes, planes, strides, output_paddings):
        super(DeResNet, self).__init__()
        assert len(planes) == len(strides)
        assert len(planes) == len(output_paddings)

        blocks = []
        for i in range(len(planes)):
            plane = planes[i]
            stride = strides[i]
            output_padding = output_paddings[i]
            block = DeResNetBlock(inplanes, plane, stride=stride, output_padding=output_padding)
            blocks.append(block)
            inplanes = plane

        self.main = nn.Sequential(*blocks)

    def init(self, x, init_scale=1.0):
        for block in self.main:
            x = block.init(x, init_scale=init_scale)
        return x

    def forward(self, x):
        return self.main(x)
