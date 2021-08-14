#
# common.py
#
# This file is part of LIDN.
#
# Jeffrey.tan<jeffrey.yf.tan@gmail.com> or <tan.y.f@163.com>
#
# Change History:
# 2020-06-20     Jeffrey.tan    the first version

import os
import torch
import numpy as np
import cv2

from argument.argument import args
from utils import log, timer, utils


def test(model, data_loader):
    info_log = log.Log(os.path.join('./experiments', args.s_experiment_name), '%(message)s')
    device = torch.device('cpu' if args.b_cpu else 'cuda')

    model.eval()

    with torch.no_grad():
        im_psnr = torch.Tensor().to(device)
        im_ssim = torch.Tensor().to(device)
        timer_test_elapsed_ticks = 0

        timer_test = timer.Timer()
        for d_index, d in enumerate(data_loader):
            t_psnr = torch.Tensor().to(device)
            t_ssim = torch.Tensor().to(device)
            for batch_index, (data, target) in enumerate(d):
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                try:
                    timer_test.restart()
                    model_out = model(data)
                    timer_test.stop()

                    timer_test_elapsed_ticks += timer_test.elapsed_ticks()
                    model_rgb = model_out[0]
                    model_rgb = model_rgb.mul(1.0).clamp(0, args.n_rgb_range)

                    all_psnr = utils.psnr(model_rgb, target, args.n_rgb_range).to(device)
                    im_psnr = torch.cat((im_psnr, all_psnr))
                    t_psnr = torch.cat((t_psnr, all_psnr))

                    out_data = model_rgb[0, :].permute(1, 2, 0).cpu().numpy()
                    out_label = target[0, :].permute(1, 2, 0).cpu().numpy()

                    if args.n_rgb_range == 255:
                        out_data = np.uint8(out_data)
                        out_label = np.uint8(out_label)
                    elif args.n_rgb_range == 65535:
                        out_data = np.uint16(out_data)
                        out_label = np.uint8(out_label)

                    all_ssim = utils.ssim(out_data, out_label, args.n_rgb_range).float().to(device)
                    im_ssim = torch.cat((im_ssim, all_ssim))
                    t_ssim = torch.cat((t_ssim, all_ssim))

                    if args.b_save_results:
                        path = os.path.join('./experiments', args.s_experiment_name, 'result',
                                            'result_' + d.dataset.name + str(batch_index) + '.png')

                        cv2.imwrite(path, cv2.cvtColor(out_data, cv2.COLOR_RGB2BGR))
                except Exception as e:
                    utils.catch_exception(e)

            t_psnr = t_psnr.mean(dim=0, keepdim=True)
            t_ssim = t_ssim.mean(dim=0, keepdim=True)

            info_log.write('{}:\t{:.3f}({:.3f}, {:.3f}, {:.3f}), {:.5f}'.format(
                d.dataset.name,
                t_psnr[:, -1].item(),
                t_psnr[:, 0].item(),
                t_psnr[:, 1].item(),
                t_psnr[:, 2].item(),
                t_ssim.item(),
            ))

        im_psnr = im_psnr.mean(dim=0, keepdim=True)
        im_ssim = im_ssim.mean(dim=0, keepdim=True)

    return im_psnr, im_ssim, timer_test_elapsed_ticks
