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


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='sg2im-models/vg128.pt')
parser.add_argument('--scene_graphs_json', default='scene_graphs/figure_6_sheep.json')
parser.add_argument('--output_dir', default='outputs')
parser.add_argument('--draw_scene_graphs', type=int, default=0)
parser.add_argument('--device', default='gpu', choices=['cpu', 'gpu'])

parser.add_argument('--use_test', action='store_true')
parser.add_argument('--test_h5', default=os.path.join('~/datasets/', 'stylized_test.h5'))
parser.add_argument('--vocab_json', default=os.path.join('~/datasets/vg/', 'vocab.json'))

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--image_size', default='64,64', type=int_tuple)
parser.add_argument('--loader_num_workers', default=2, type=int)
parser.add_argument('--include_relationships', default=True, type=bool_flag)
parser.add_argument('--max_objects_per_image', default=10, type=int)
parser.add_argument('--vg_use_orphaned_objects', default=True, type=bool_flag)


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
  model = Sg2ImModel(**checkpoint['model_kwargs'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()
  model.to(device)

  if args.use_test:
    with open(args.vocab_json, 'r') as f:
      vocab = json.load(f)

    dset_kwargs = {
      'vocab': vocab,
      'h5_path': args.test_h5,
      'image_dir': args.vg_image_dir,
      'image_size': args.image_size,
      'max_samples': None,
      'max_objects': args.max_objects_per_image,
      'use_orphaned_objects': args.vg_use_orphaned_objects,
      'include_relationships': args.include_relationships,
      'stylized_dir': None
    }
    test_dset = VgSceneGraphDataset(**test_kwargs)
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
      print("Batch", idx)
        
      masks = None
      batch = [batch[i].to(device) for i in range(1, len(batch))]
      if len(batch) == 7:
        filenames, style_ids, objs, boxes, triples, obj_to_img, triple_to_img = batch
      elif len(batch) == 8:
        filenames, style_ids, objs, boxes, masks, triples, obj_to_img, triple_to_img = batch
      else:
        assert False
      predicates = triples[:, 1]
        
      model_boxes = boxes
      model_masks = masks 
      with torch.no_grad():
        model_out = model(objs, triples, obj_to_img, boxes_gt=model_boxes, 
                          masks_gt=model_masks, style_batch=style_ids)
        imgs, boxes_pred, masks_pred, predicate_scores = model_out
        imgs = imagenet_deprocess_batch(imgs)
        
      for i in range(imgs.shape[0]):
        img_np = imgs[i].cpu().numpy().transpose(1, 2, 0)
        img_path = os.path.join(args.output_dir, filenames[i])
        imwrite(img_path, img_np)
  else:
    # Load the scene graphs
    with open(args.scene_graphs_json, 'r') as f:
    scene_graphs = json.load(f)

    # Run the model forward
    with torch.no_grad():
    imgs, boxes_pred, masks_pred, _ = model.forward_json(scene_graphs)
    imgs = imagenet_deprocess_batch(imgs)

    # Save the generated images
    for i in range(imgs.shape[0]):
    img_np = imgs[i].numpy().transpose(1, 2, 0)
    img_path = os.path.join(args.output_dir, 'img%06d.png' % i)
    imwrite(img_path, img_np)

    # Draw the scene graphs
    if args.draw_scene_graphs == 1:
    for i, sg in enumerate(scene_graphs):
      sg_img = vis.draw_scene_graph(sg['objects'], sg['relationships'])
      sg_img_path = os.path.join(args.output_dir, 'sg%06d.png' % i)
      imwrite(sg_img_path, sg_img)


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

