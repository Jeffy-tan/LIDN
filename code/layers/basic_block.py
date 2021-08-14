#
# basic_block.py
#
# This file is part of LIDN.
#
# Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import torch.nn as nn
import numpy as np


class ConvReLUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False, bn=True, af=None, dilation=1, groups=1):
        super(ConvReLUBlock, self).__init__()

        pd = kernel_size // 2 if type(kernel_size) == int else tuple(np.array(kernel_size) // 2)

        self.layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pd, bias=bias, dilation=dilation, groups=groups)]
        if bn:
            self.layers.append(nn.BatchNorm2d(out_channels))
        if af is not None:
            self.layers.append(af)

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        y = self.layers(x)
        return y

