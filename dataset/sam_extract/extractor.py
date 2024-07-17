import torch
import cv2
from sam import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import os
import argparse
from glob import glob
import pickle

sam_checkpoint = "./sam_extract/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.92,
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,
)

import numpy as np
import matplotlib.pyplot as plt

def get_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 3))
    for ann in sorted_anns:
        m = ann['segmentation']
        img[m] = img[m] * 0.80 + np.concatenate([np.random.random(3)])*0.2
    return img

def single_extractor(pic_name, out_pic_name):
    image = cv2.imread(pic_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    masks = mask_generator.generate(image)
    print(len(masks))
    print(masks[0].keys())
    anns_image = get_anns(masks)
    print(f'Anns_image : {anns_image.shape}, ({anns_image.min()},{anns_image.max()})')
    cv2.imwrite(out_pic_name, (anns_image*255).astype(int))
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SAM Segmentation extraction.')
    parser.add_argument('--split_file', default=None)
    parser.add_argument('--image_path', type=str, required=True, help='path of the extracted image.')
    parser.add_argument('--output_path', type=str, required=True, help='path to file containing extracted descriptors.')
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    image_list = sorted(glob(os.path.join(args.image_path, "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))
    print(image_list[:10])
    if args.split_file is not None:
        with open(args.split_file, 'rb') as file:
            serialized_list = file.read()
            train_list, eval_list = pickle.loads(serialized_list)
        print(train_list)
        
    for image in image_list:
        if args.split_file is not None:
            if os.path.basename(image) not in train_list:
                continue
        output_name = os.path.join(args.output_path, os.path.basename(image))
        
        print('Process : ', image, ' -> ', output_name)
        single_extractor(image, output_name)