#
# channel_attention.py
#
# This file is part of LIDN.
#
# Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import torch.nn as nn


class CABlock(nn.Module):
    def __init__(self, in_channel, reduction=1):
        super(CABlock, self).__init__()

        out_channel = in_channel

        self.ca_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // reduction, out_channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.ca_block(x)
        return x * y
