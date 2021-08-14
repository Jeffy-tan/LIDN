#
# model_base.py
#
# This file is part of LIDN.
#
# Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import torch
import torch.nn as nn


class ModelBase(nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()

        self.args = args

        self.device = torch.device('cpu' if args.b_cpu else 'cuda')
        self.n_gpu = args.n_gpu

    def forward(self, x):
        pass
