#
# wed.py
#
# This file is part of LIDN.
#
# Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import torch

import numpy as np

from data import data_base


class Wed(data_base.DataBase):
    def __init__(self, args, b_train):
        dataset_name = 'wed'
        dataset_range = '1-150912/1-100'  # '1-150912/1-100'
        dataset_ext = ('bmp',)

        super(Wed, self).__init__(args, dataset_name, dataset_range, dataset_ext, b_train)

    def __getitem__(self, index):
        im_data = 0
        im_label = 0
        if self.data_pack == 'packet':
            im_data = self.bin_data[index]
            im_label = self.bin_label[index]
        elif self.data_pack == 'bin':
            im_data = np.load(self.bin_data[index])
            im_label = np.load(self.bin_label[index])
        else:
            pass

        return torch.from_numpy(im_data / 1.0).float(), torch.from_numpy(im_label / 1.0).float()
