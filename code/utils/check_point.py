#
# check_point.py
#
# This file is part of LIDN.
#
# Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import datetime
import os
import matplotlib.pyplot as plt
import torch

plt.switch_backend('agg')


class CheckPoint(object):
    def __init__(self, filepath, experiment_name):
        self.loss = torch.Tensor()
        self.result = torch.Tensor()
        self.experiment_name = experiment_name
        self.start_epoch = 0
        self.color = ('red', 'green', 'blue', 'black')
        self.psnr_label = ('R_PSNR', 'G_PSNR', 'B_PSNR', 'CPSNR')
        self.label = 'result on {}'.format(self.experiment_name)

        self.filepath = os.path.join(filepath, self.experiment_name)
        os.makedirs(self.filepath, exist_ok=True)
        os.makedirs(os.path.join(self.filepath, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self.filepath, 'result'), exist_ok=True)

    def load(self, to_device, is_best=False):
        model_name = 'model_best.pth' if is_best else 'model_latest.pth'
        return torch.load(os.path.join(self.filepath, 'model', model_name), map_location=to_device)

    def save_config(self, args, model):
        with open(os.path.join(self.filepath, 'config.txt'), 'w') as f:
            f.write(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '\n\n')
            print(model, file=f)
            f.write('\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
