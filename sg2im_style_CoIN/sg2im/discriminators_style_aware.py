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

import torch
import torch.nn as nn
import torch.nn.functional as F

from sg2im.bilinear import crop_bbox_batch
from sg2im.layers import GlobalAvgPool, Flatten, get_activation, build_cnn, styleAwareSequential


class StyleAwarePatchDiscriminator(nn.Module):
  def __init__(self, arch, normalization='conditional', activation='leakyrelu-0.2',
               padding='same', pooling='avg', input_size=(128,128),
               layout_dim=0):
    super(StyleAwarePatchDiscriminator, self).__init__()
    input_dim = 3 + layout_dim
    arch = 'I%d,%s' % (input_dim, arch)
    cnn_kwargs = {
      'arch': arch,
      'normalization': 'conditional',
      'activation': activation,
      'pooling': pooling,
      'padding': padding,
    }
    self.cnn, output_dim = build_cnn(**cnn_kwargs)
    self.classifier = nn.Conv2d(output_dim, 1, kernel_size=1, stride=1) # TODO: is this thing ever used? Try commenting it out.

  def forward(self, x, style_batch, layout=None):
    if layout is not None:
      x = torch.cat([x, layout], dim=1)
    return self.cnn(x, style_batch)


class StyleAwareAcDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='conditional', activation='relu',
               padding='same', pooling='avg'):
    super(StyleAwareAcDiscriminator, self).__init__()
    self.vocab = vocab

    cnn_kwargs = {
      'arch': arch,
      'normalization': 'conditional',
      'activation': activation,
      'pooling': pooling, 
      'padding': padding,
    }
    cnn, D = build_cnn(**cnn_kwargs)
    self.cnn = styleAwareSequential(cnn, GlobalAvgPool(), nn.Linear(D, 1024))
    num_objects = len(vocab['object_idx_to_name'])

    self.real_classifier = nn.Linear(1024, 1)
    self.obj_classifier = nn.Linear(1024, num_objects)

  def forward(self, x, y, style_batch):
    if x.dim() == 3:
      x = x[:, None]
    vecs = self.cnn(x, style_batch)
    real_scores = self.real_classifier(vecs)
    obj_scores = self.obj_classifier(vecs)
    ac_loss = F.cross_entropy(obj_scores, y)
    return real_scores, ac_loss


class StyleAwareAcCropDiscriminator(nn.Module):
  def __init__(self, vocab, arch, normalization='conditional', activation='relu',
               object_size=64, padding='same', pooling='avg'):
    super(StyleAwareAcCropDiscriminator, self).__init__()
    self.vocab = vocab
    self.discriminator = StyleAwareAcDiscriminator(vocab, arch, normalization,
                                         activation, padding, pooling)
    self.object_size = object_size

  def forward(self, imgs, objs, boxes, obj_to_img, style_batch):
    crops, style_batch_B = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size, style_batch=style_batch)
    real_scores, ac_loss = self.discriminator(crops, objs, style_batch_B)
    return real_scores, ac_loss
