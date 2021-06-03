#!/usr/bin/python
#
# Copyright 2018 Google LLC
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

import os
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import h5py
import PIL

from .utils import imagenet_preprocess, Resize


class VgEvalDataset(Dataset):
  def __init__(self, ground_truth_dir, model_out_dir, image_size=(64, 64), name_fn=None):
    super(VgEvalDataset, self).__init__()
    
    self.ground_truth_dir = ground_truth_dir
    self.model_out_dir = model_out_dir
    self.image_size = image_size
    self.model_out_imgs = os.listdir(model_out_dir)
    self.ground_truth_imgs = os.listdir(ground_truth_dir)

    transform = [Resize(image_size), T.ToTensor()]
    self.transform = T.Compose(transform)
    self.name_fn = name_fn

  def __len__(self):
    return len(self.model_out_imgs)

  def __getitem__(self, index):
    """
    Returns a tuple of:
    - image: FloatTensor of shape (C, H, W)
    """  
    filename = self.model_out_imgs[index]
    model_out_img = os.path.join(self.model_out_dir, filename)
    if self.name_fn is not None:
        filename = self.name_fn(filename)
    ground_truth_img = os.path.join(self.ground_truth_dir, filename)
    
    with open(ground_truth_img, 'rb') as f:
        with PIL.Image.open(f) as image:
            ground_truth_img = self.transform(image.convert('RGB'))
   
    with open(model_out_img, 'rb') as f:
        with PIL.Image.open(f) as image:
            model_out_img = self.transform(image.convert('RGB'))

    return ground_truth_img, model_out_img


def vg_collate_fn(batch):
  """
  Collate function to be used when wrapping a VgSceneGraphDataset in a
  DataLoader. Returns a tuple of the following:

  - imgs: FloatTensor of shape (N, C, H, W)
  """
  # batch is a list, and each element is (image, objs, boxes, triples) 
  all_ground_truth_imgs, all_model_out_imgs = [], []
  for i, (ground_truth_img, model_out_img) in enumerate(batch):
    all_ground_truth_imgs.append(ground_truth_img[None])
    all_model_out_imgs.append(model_out_img[None])
    
  all_ground_truth_imgs = torch.cat(all_ground_truth_imgs)
  all_model_out_imgs = torch.cat(all_model_out_imgs)

  out = (all_ground_truth_imgs, all_model_out_imgs)
  return out


def vg_uncollate_fn(batch):
  """
  Inverse operation to the above.
  """
  all_ground_truth_imgs, all_model_out_imgs = batch
  out = []
  for i in range(all_ground_truth_imgs.size(0)):
    cur_ground_truth_img = all_ground_truth_imgs[i]
    cur_model_out_img = all_model_out_imgs[i]
    out.append((cur_ground_truth_img, cur_model_out_img))
  return out

