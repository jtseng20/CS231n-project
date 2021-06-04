#!/usr/bin/python
#
# Copyright 2020 Helisa Dhamo, Iro Laina
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import os
import yaml
from addict import Dict
from collections import defaultdict

import pickle
import random
import pytorch_ssim
import argparse

from imageio import imsave
import sys
sys.path.append('./.')
from sg2im.data.utils import imagenet_deprocess_batch
from sg2im.data.vg_style_eval import VgEvalDataset, vg_collate_fn
from torch.utils.data import DataLoader

import lpips


parser = argparse.ArgumentParser()
parser.add_argument('--loader_num_workers', default=1, type=int)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--ground_truth_dir', default='/vision2/u/helenav/datasets/vg_style')
parser.add_argument('--model_results_dir', default='/scr/helenav/outputs/style_test_sg2im_w_conditional_norm_both')

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--splice_name', action='store_true')

args = parser.parse_args()


def main():
    device = torch.device('cuda:0')
    print(os.listdir(args.ground_truth_dir)[0])
    print(os.listdir(args.model_results_dir)[0])
    if args.splice_name:
        name_fn = lambda x: x[9:]
    else:
        name_fn = None
    dset = VgEvalDataset(ground_truth_dir=args.ground_truth_dir, model_out_dir=args.model_results_dir, name_fn=name_fn)
    loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': args.shuffle,
    'collate_fn': vg_collate_fn,
     }
    loader = DataLoader(dset, **loader_kwargs)
    eval_model(loader, device=device)

def eval_model(loader, device):
    num_batches = 0
    num_samples = 0
    mae_per_image = []
    ssim_per_image = []
    margin = 2

    ## Initializing the perceptual loss model
    lpips_model = lpips.LPIPS(net='alex').to(device)
    perceptual_error_image = []
    # ---------------------------------------

    img_idx = 0

    with torch.no_grad():
        for batch in loader:
            # Loads in ground truth first and model output second
            num_batches += 1
            batch = [tensor.to(device) for tensor in batch]

            imgs, imgs_pred = batch

            num_samples += imgs.shape[0]
            imgs = imagenet_deprocess_batch(imgs).float()
            imgs_pred = imagenet_deprocess_batch(imgs_pred).float()

            # MAE per image
            mae_per_image.append(torch.mean(
                torch.abs(imgs - imgs_pred).view(imgs.shape[0], -1), 1).cpu().numpy())
            for s in range(imgs.size(0)):
                ssim_per_image.append(
                        pytorch_ssim.ssim(imgs[s:s+1, :, :, :] / 255.0,
                                          imgs_pred[s:s+1, :, :, :] / 255.0, window_size=3).cpu().item())

                # normalize as expected from the LPIPS model
                imgs_pred_norm = imgs_pred[s:s+1, :, :, :] / 127.5 - 1
                imgs_gt_norm = imgs[s:s+1, :, :, :] / 127.5 - 1
                perceptual_error_image.append(
                        lpips_model(imgs_pred_norm.to(device), imgs_gt_norm.to(device)).detach().cpu().numpy())

            if num_batches % 10 == 0:
                calculate_scores(mae_per_image, ssim_per_image, perceptual_error_image)

            if num_batches % 10 == 0:
                save_results(mae_per_image, ssim_per_image, perceptual_error_image, 'final')
            
            print(args.ground_truth_dir)
            print(args.model_results_dir)
            print(img_idx)
            img_idx += 1

    calculate_scores(mae_per_image, ssim_per_image, perceptual_error_image)
    save_results(mae_per_image, ssim_per_image, perceptual_error_image, 'final')


def calculate_scores(mae_per_image, ssim_per_image, perceptual_error_image):

    mae_all = np.mean(np.hstack(mae_per_image), dtype=np.float64)
    mae_std = np.std(np.hstack(mae_per_image), dtype=np.float64)
    ssim_all = np.mean(ssim_per_image, dtype=np.float64)
    ssim_std = np.std(ssim_per_image, dtype=np.float64)
    # percept error -----------
    percept_all = np.mean(perceptual_error_image, dtype=np.float64)
    percept_all_std = np.std(perceptual_error_image, dtype=np.float64)
    # ------------------------

    print()
    print('MAE: Mean {:.6f}, Std {:.6f}'.format(mae_all, mae_std))
    print('SSIM: Mean {:.6f}, Std {:.6f}'.format(ssim_all, ssim_std))
    print('LPIPS: Mean {:.6f}, Std {:.6f}'.format(percept_all, percept_all_std))


def save_results(mae_per_image, ssim_per_image, perceptual_error_image, iter):

    results = dict()
    results['mae_per_image'] = mae_per_image
    results['ssim_per_image'] = ssim_per_image
    results['perceptual_per_image'] = perceptual_error_image

if __name__ == '__main__':
    main()
