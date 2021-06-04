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

import argparse, json, os

from imageio import imwrite
import torch
import sys
sys.path.append('./.')
from sg2im.model import Sg2ImModel
from sg2im.data.utils import imagenet_deprocess_batch
import sg2im.vis as vis
from PIL import Image
import numpy as np
from sg2im.data.utils import imagenet_preprocess, Resize, linear_interp
import torchvision.transforms as T
from sg2im.data.vg_style_inject import VgSceneGraphDataset, vg_collate_fn


import functools
import os
import json
import math
from collections import defaultdict
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sg2im.discriminators import PatchDiscriminator, AcCropDiscriminator
from sg2im.losses import get_gan_losses
from sg2im.metrics import jaccard
from sg2im.utils import int_tuple, float_tuple, str_tuple
from sg2im.utils import timeit, bool_flag, LossManager

parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint', default='/scr/helenav/checkpoints_simsg/style_weight_100/checkpoint_with_model.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_street.json')
#/scr/helenav/outputs/style_test_inject_weight_100
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])
parser.add_argument('--style_image', default='/vision2/u/helenav/datasets/style-images/')

# Interpolation arguments
parser.add_argument('--do_interp_test', dest='do_interp_test', default=False, action='store_true')
parser.add_argument('--latent_path', default='/vision2/u/helenav/datasets/style-images/')
parser.add_argument('--latent_1', default='6.jpg')
parser.add_argument('--latent_2', default='22.jpg')
parser.add_argument('--interp_steps', type=int, default=8)

# Noise injection
parser.add_argument('--inject_noise', dest='inject_noise', default=False, action='store_true')

parser.add_argument('--use_test', action='store_true')
parser.add_argument('--test_h5', default=os.path.join('/scr/helenav/datasets/preprocess_vg', 'stylized_test.h5'))
parser.add_argument('--vocab_json', default=os.path.join('/scr/helenav/datasets/preprocess_vg', 'vocab.json'))
parser.add_argument('--vg_image_dir', default='/vision2/u/helenav/datasets/vg/images')

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--image_size', default='64,64',type=int_tuple)
parser.add_argument('--loader_num_workers', default=2, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)


def do_interp_test(args, model, device):
  print("Doing interpolation test")
  def process_img(path):
    img = Image.open(args.latent_path+path)
    transform = T.Compose([Resize((64,64)), T.ToTensor(), imagenet_preprocess()])
    return torch.unsqueeze(transform(img.convert('RGB')),0).to(device)
  
  latent_1 = model.style_map(process_img(args.latent_1))
  latent_2 = model.style_map(process_img(args.latent_2))
  interp_list = linear_interp(latent_1, latent_2, args.interp_steps)
  
  for num, interp in enumerate(interp_list):
    name = "interp" + args.latent_1[:-4] + "_" + args.latent_2[:-4] + "_" + str(num) + "_"
    with open(args.scene_graphs_json, 'r') as f:
      scene_graphs = json.load(f)  

    interp = torch.unsqueeze(interp,0).to(device)
    # Run the model forward
    with torch.no_grad():
      imgs, boxes_pred, masks_pred, _ = model.forward_json_manual_latent(scene_graphs, style_encoding=interp)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated images
    for i in range(imgs.shape[0]):
      img_np = imgs[i].numpy().transpose(1, 2, 0)
      img_path = os.path.join(args.output_dir, name + 'img%06d.png' % i)
      imwrite(img_path, img_np)
    
def run_test_set(args, model, device):
  with open(args.vocab_json, 'r') as f:
    vocab = json.load(f)

  dset_kwargs = {
    'vocab': vocab,
    'h5_path': args.test_h5,
    'image_dir': args.vg_image_dir,
    'style_reference_dir': args.style_image,
    'image_size': args.image_size,
    'max_samples': None,
    'max_objects': args.max_objects_per_image,
    'use_orphaned_objects': args.vg_use_orphaned_objects,
    'include_relationships': args.include_relationships,
    'stylized_dir': args.style_image,
    'testing':True
      
  }
  test_dset = VgSceneGraphDataset(**dset_kwargs)
  iter_per_epoch = len(test_dset) // args.batch_size
  print('There are %d iterations per epoch' % iter_per_epoch)

  loader_kwargs = {
    'batch_size': args.batch_size,
    'num_workers': args.loader_num_workers,
    'shuffle': False,
    'collate_fn': vg_collate_fn,
  }
  test_loader = DataLoader(test_dset, **loader_kwargs)
  
  for idx, batch in enumerate(test_loader):
    if idx % 100 == 0:
      print(idx)
      
    masks = None
    batch = [batch[0]] + [batch[i].to(device) for i in range(1, len(batch))]
    print(len(batch))
    stylized_output_file, style_img, style_id, objs, boxes, triples, obj_to_img, triple_to_img = batch
    predicates = triples[:, 1]
      
    model_boxes = boxes
    model_masks = masks 
    with torch.no_grad():
      model_out = model(objs=objs, triples=triples, obj_to_img=obj_to_img,
            boxes_gt=boxes, masks_gt=None, style_img=style_img)
      imgs, boxes_pred, masks_pred, predicate_scores = model_out
      imgs = imagenet_deprocess_batch(imgs)
      
    for i in range(imgs.shape[0]):
      img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
      img_path = os.path.join(args.output_dir, stylized_output_file[i])
      imwrite(img_path, img_np)
    print(os.listdir(args.output_dir))
    

def main(args):
  if not os.path.isfile(args.checkpoint):
    print('ERROR: Checkpoint file "%s" not found' % args.checkpoint)
    print('Maybe you forgot to download pretraind models? Try running:')
    print('bash scripts/download_models.sh')
    return

  if not os.path.isdir(args.output_dir):
    print('Output directory "%s" does not exist; creating it' % args.output_dir)
    os.makedirs(args.output_dir)

  if args.device == 'cpu':
    device = torch.device('cpu')
  elif args.device == 'gpu':
    device = torch.device('cuda:0')
    if not torch.cuda.is_available():
      print('WARNING: CUDA not available; falling back to CPU')
      device = torch.device('cpu')
        
  # Load the model, with a bit of care in case there are no GPUs
  map_location = 'cpu' if device == torch.device('cpu') else None
  checkpoint = torch.load(args.checkpoint, map_location=map_location)   

  checkpoint['model_kwargs']['num_stylish'] = 0
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  print(model)
  model.eval()
  model.to(device)
  
  if args.do_interp_test:
    do_interp_test(args, model, device)
    return
  elif args.use_test:
    print("running test set")
    run_test_set(args, model, device)
    return
  
  style_images = [f for f in os.listdir(args.style_image)]
  print(style_images)
  for style_image in style_images:
    name = "_" + "style" + style_image[:-4]
    style_image = Image.open(args.style_image + style_image)
    with open(args.scene_graphs_json, 'r') as f:
      scene_graphs = json.load(f)
    transform = [Resize((64,64)), T.ToTensor(), imagenet_preprocess()]
    transform = T.Compose(transform)
    style_image = transform(style_image.convert('RGB')) 
    style_image = torch.unsqueeze(style_image, 0)  

    style_image = style_image.to(device)
    if args.inject_noise:
      style_image = torch.randn_like(style_image)
    # Run the model forward
    with torch.no_grad():
      imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graphs, style_img=style_image)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated images
    for i in range(imgs.shape[0]):
      img_np = imgs[i].numpy().transpose(1, 2, 0)
      img_path = os.path.join(args.output_dir, f'img00000{i}' + name + '.png')
      imwrite(img_path, img_np)

    # Draw the scene graphs
    if args.draw_scene_graphs == 1:
      for i, sg in enumerate(scene_graphs):
        sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
        sg_img_path = os.path.join(args.output_dir, 'sg%06d.png' % i)
        imwrite(sg_img_path, sg_img)
    
    if args.inject_noise:
      break


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

