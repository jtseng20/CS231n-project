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
               normalization='instance', activation='leakyrelu', add_style_channel=False):
    super(RefinementModule, self).__init__()
    
    layers = []
    layers.append(nn.Conv2d(layout_dim + input_dim + add_style_channel, output_dim,
                            kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization))
    layers.append(get_activation(activation))
    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization))
    layers.append(get_activation(activation))
    layers = [layer for layer in layers if layer is not None]
    for layer in layers:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
    self.net = nn.Sequential(*layers)

  def forward(self, layout, feats, style_vector=None):
    _, _, HH, WW = layout.size()
    _, _, H, W = feats.size()
    assert HH >= H
    if HH > H:
      factor = round(HH // H)
      assert HH % factor == 0
      assert WW % factor == 0 and WW // factor == W
      layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)
    if style_vector is None:
      net_input = torch.cat([layout, feats], dim=1)
    else:
      net_input = torch.cat([layout, feats, style_vector], dim=1)
    out = self.net(net_input)
    return out


class RefinementNetwork(nn.Module):
  def __init__(self, dims, normalization='instance', activation='leakyrelu'):
    super(RefinementNetwork, self).__init__()
    layout_dim = dims[0]
    self.refinement_modules = nn.ModuleList()
    for i in range(1, len(dims)):
      input_dim = 1 if i == 1 else dims[i - 1]
      output_dim = dims[i]
      mod = RefinementModule(layout_dim, input_dim, output_dim,
                             normalization=normalization, activation=activation, add_style_channel=(i==1))
      self.refinement_modules.append(mod)
    output_conv_layers = [
      nn.Conv2d(dims[-1], dims[-1], kernel_size=3, padding=1),
      get_activation(activation),
      nn.Conv2d(dims[-1], 3, kernel_size=1, padding=0)
    ]
    nn.init.kaiming_normal_(output_conv_layers[0].weight)
    nn.init.kaiming_normal_(output_conv_layers[2].weight)
    self.output_conv = nn.Sequential(*output_conv_layers)
    torch.random.manual_seed(69)
    self.style_num = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.style_dict = torch.rand((self.style_num, 1, 4, 4)).to(device)
    assert torch.sum(self.style_dict) - 229.9530 < 1e-5
    
  def forward(self, layout, style_batch=None):
    """
    Output will have same size as layout
    
    style_vector will be (N,) where every element at N_i is between 0 and self.style_num
    """
    # H, W = self.output_size
    N, _, H, W = layout.size()
    self.layout = layout
    # Figure out size of input
    input_H, input_W = H, W
    for _ in range(len(self.refinement_modules)):
      input_H //= 2
      input_W //= 2

    assert input_H != 0
    assert input_W != 0

    feats = torch.zeros(N, 1, input_H, input_W).to(layout)
    for idx, mod in enumerate(self.refinement_modules):
      feats = F.upsample(feats, scale_factor=2, mode='nearest')
      if idx == 0:
        feats = mod(layout, feats, style_vector=self.style_dict[style_batch])
      else:
        feats = mod(layout, feats)

    out = self.output_conv(feats)
    return out
