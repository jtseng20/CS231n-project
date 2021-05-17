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

from sg2im.layers import get_normalization_2d
from sg2im.layers import get_activation
from sg2im.utils import timeit, lineno, get_gpu_memory


"""
Cascaded refinement network architecture, as described in:

Qifeng Chen and Vladlen Koltun,
"Photographic Image Synthesis with Cascaded Refinement Networks",
ICCV 2017
"""


class RefinementModule(nn.Module):
  def __init__(self, layout_dim, input_dim, output_dim,
               normalization='instance', activation='leakyrelu', style_dim=None):
    super(RefinementModule, self).__init__()
    layers = []
    self.norm_levels = [1, 4]
    layers.append(nn.Conv2d(layout_dim + input_dim, output_dim,
                            kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization=normalization, style_dim=style_dim))
    layers.append(get_activation(activation))
    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization=normalization, style_dim=style_dim))
    layers.append(get_activation(activation))
    layers = [layer.cuda() for layer in layers if layer is not None]
    for layer in layers:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
    self.layers = layers

  def forward(self, layout, feats, style=None):
    _, _, HH, WW = layout.size()
    _, _, H, W = feats.size()
    assert HH >= H
    if HH > H:
      factor = round(HH // H)
      assert HH % factor == 0
      assert WW % factor == 0 and WW // factor == W
      layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
    net_input = torch.cat([layout, feats], dim=1)
    out = net_input.clone()
    for idx, layer in enumerate(self.layers):
        if idx in self.norm_levels:
            out = layer(out, style)
        else:
            out = layer(out)
    return out


class RefinementNetwork(nn.Module):
  def __init__(self, dims, normalization='instance', activation='leakyrelu', style_dim=None, num_stylish=2):
    super(RefinementNetwork, self).__init__()
    layout_dim = dims[0]
    self.refinement_modules = nn.ModuleList()
    for i in range(1, len(dims)):
      normalization = 'stylish' if i <= num_stylish else 'instance'
      input_dim = 1 if i == 1 else dims[i - 1]
      output_dim = dims[i]
      mod = RefinementModule(layout_dim, input_dim, output_dim,
                             normalization=normalization, activation=activation, style_dim=style_dim)
      self.refinement_modules.append(mod)
    output_conv_layers = [
      nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
      get_activation(activation),
      nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
    ]
    nn.init.kaiming_normal_(output_conv_layers[0].weight)
    nn.init.kaiming_normal_(output_conv_layers[2].weight)
    self.output_conv = nn.Sequential(*output_conv_layers)

  def forward(self, layout, style=None):
    """
    Output will have same size as layout
    """
    # H, W = self.output_size
    N, _, H, W = layout.size()
    self.layout = layout
    self.style = style

    # Figure out size of input
    input_H, input_W = H, W
    for _ in range(len(self.refinement_modules)):
      input_H //= 2
      input_W //= 2

    assert input_H != 0
    assert input_W != 0

    feats = torch.zeros(N, 1, input_H, input_W).to(layout)
    for mod in self.refinement_modules:
      feats = F.upsample(feats, scale_factor=2, mode='nearest')
      feats = mod(layout, feats, style)

    out = self.output_conv(feats)
    return out