import h5py
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--h5_path', default='/scr/helenav/datasets/preprocess_vg/test.h5')
parser.add_argument('--stylized_dir', default= '/vision2/u/helenav/datasets/vg_style')
parser.add_argument('--vg_dir', default= '/vision2/u/helenav/datasets/vg')
parser.add_argument('--output_h5_path', default= '/scr/helenav/datasets/preprocess_vg/stylized_test.h5')
args = parser.parse_args()

def run_script():
    style_imgs = [(int(f.split("_style")[0]), int(f.split("_style")[1][:-4])) for f in os.listdir(args.stylized_dir)]
    style_imgs = sorted(style_imgs, key = lambda x: x[1])
    style_imgs = sorted(style_imgs, key = lambda x: x[0])
    style_vg, style_id = zip(*style_imgs)
    output_h5 = h5py.File(args.output_h5_path, 'w')
    original_h5 = h5py.File(args.h5_path, 'r')
    original_dict = {k:np.asarray(v) for k, v in original_h5.items() if k != 'object_attributes'}
    output_dict = {}
    style_vg = set(style_vg)
    print("Finished preprocess...")
    indices = []
    for idx, img_id in enumerate(original_h5['image_ids']):
        print(idx)
        if img_id in style_vg:
            indices.append(idx)
    for k, v in original_dict.items():
        if k != 'object_attributes':
            output_dict[k] = v[indices]
    for k, v in output_dict.items():
        output_h5.create_dataset(k, data=v)
    output_h5.create_dataset('object_attributes', data=original_h5['object_attributes'])
    print("Finished copying...")
    
    # Assumes that all images have the same number of styles! 
    output_h5.create_dataset('style_ids', data=sorted(list(set(style_id))))
    for k,v in output_h5.items():
        print(k, v)
    print("Finished checking...")

if __name__ == '__main__':
    run_script()