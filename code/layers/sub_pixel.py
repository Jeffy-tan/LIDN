#
# sub_pixel.py
#
# This file is part of LIDN.
#
# Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import torch
import torch.nn as nn
from typing import Tuple


class SubPixelConv(nn.Module):
    def __init__(self, n_channels=64, upsample=2):
        super(SubPixelConv, self).__init__()
        self.n_channels = n_channels
        self.upsample = upsample
        self.out_channels = self.upsample * self.upsample * self.n_channels

        self.conv = nn.Conv2d(in_channels=self.n_channels,
                              out_channels=self.out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=False)
        self.upsample_net = nn.PixelShuffle(self.upsample)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input = x
        output = self.relu(self.upsample_net(self.conv(x)))
        return output


def pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
    ph = factor_hw[0]
    pw = factor_hw[1]
    y = x
    b, ic, ih, iw = y.shape
    oc, oh, ow = ic // (ph * pw), ih * ph, iw * pw
    y = y.reshape(b, oc, ph, pw, ih, iw)
    y = y.permute(0, 1, 4, 2, 5, 3)     # B, oC, iH, pH, iW, pW
    y = y.reshape(b, oc, oh, ow)
    return y


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixelshuffle(x, self.upscale_factor)


def pixelshuffle_invert(x: torch.Tensor, factor_hw: Tuple[int, int]):
    ph = factor_hw[0]
    pw = factor_hw[1]
    y = x
    b, ic, ih, iw = y.shape
    oc, oh, ow = ic * (ph * pw), ih // ph, iw // pw
    y = y.reshape(b, ic, oh, ph, ow, pw)
    y = y.permute(0, 1, 3, 5, 2, 4)     # B, iC, pH, pW, oH, oW
    y = y.reshape(b, oc, oh, ow)
    return y


class PixelShuffleInvert(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffleInvert, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixelshuffle_invert(x, self.upscale_factor)
