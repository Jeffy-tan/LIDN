#
# argument.py
#
# This file is part of LIDN.
#
# Copyright (c) 2020 Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import argparse

parser = argparse.ArgumentParser(description='template of demosaick')

# Hardware specifications
parser.add_argument('--b_cpu', type=bool, default=True,
                    help='use cpu only')
parser.add_argument('--n_gpu', type=int, default=1,
                    help='number of GPU')
parser.add_argument('--b_cudnn', type=bool, default=True,
                    help='use cudnn')
parser.add_argument('--n_seed', type=int, default=1,
                    help='random seed')

# Model specifications
parser.add_argument('--s_model', '-m', default='lidn.LIDN',  # 'lidn_basic.LIDNbasic'
                    help='model name')
parser.add_argument('--b_load_best', type=bool, default=True,
                    help='use best model for testing')

# Data specifications
parser.add_argument('--dir_dataset', type=str, default='../DATA',
                    help='dataset directory')
parser.add_argument('--n_patch_size', type=int, default=100,
                    help='output patch size')
parser.add_argument('--n_rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--data_pack', type=str, default='packet/packet',
                    choices=('packet', 'bin', 'ori'),
                    help='make binary data')

# Evaluation specifications
parser.add_argument('--s_eval_dataset', '-e', default='mcm.Mcm+kodak.Kodak',
                    help='evaluation dataset')

# Log specifications
parser.add_argument('--s_experiment_name', type=str, default='LIDN',  # 'LIDN-basic'
                    help='file name to save')
parser.add_argument('--b_save_results', type=bool, default=True,
                    help='save output results')

args = parser.parse_args()

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False
