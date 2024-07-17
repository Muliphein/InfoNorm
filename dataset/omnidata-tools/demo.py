# adapted from https://github.com/EPFL-VILAB/omnidata
import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys
import cv2 as cv


parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models", default='./pretrained_models/')

parser.add_argument('--task', dest='task', help="normal or depth")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = args.pretrained_models 

from modules.unet import UNet
from modules.midas.dpt_depth import DPTDepthModel
from data.transforms import get_transform

trans_topil = transforms.ToPILImage()
os.system(f"mkdir -p {args.output_path}")
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if args.task == 'normal':
    image_size = 384
    
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        get_transform('rgb', image_size=None)])

elif args.task == 'depth':
    image_size = 384
    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384')
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose([
                                transforms.CenterCrop((360, 360)),
                                ])


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    img = torch.nan_to_num(img, nan=trunc_mean)
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}.png')

        # print(f'Reading input {img_path} ...')
        img = Image.open(img_path)

        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        rgb_path = os.path.join(args.output_path, f'{output_file_name}_rgb.png')
        trans_rgb(img).save(rgb_path)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            output = output.clamp(0,1)
            output_npz = output.detach().cpu().numpy()[0]
            # print(f'Before Np : {output_npz.shape} ({output_npz.min()},{output_npz.max()})')
            output_npz = cv.resize(output_npz, (360, 360))
            # print(f'After Np : {output_npz.shape} ({output_npz.min()},{output_npz.max()})')
            np.save(save_path.replace('.png', '_depth.npy'), output_npz)
            # plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
        else:
            output_npz = output.detach().cpu().numpy()[0]
            # print(f'Before Np : {output_npz.shape} ({output_npz.min()},{output_npz.max()})')
            output_npz = output_npz.transpose((1, 2, 0))
            output_npz = cv.resize(output_npz, (360, 360))
            output_npz = output_npz.transpose((2, 0, 1))
            # print(f'After Np : {output_npz.shape} ({output_npz.min()},{output_npz.max()})')
            np.save(save_path.replace('.png', '_normal.npy'), output_npz)
            # trans_topil(output[0]).save(save_path)
            
        # print(f'Writing output {save_path} ...')


img_path = Path(args.img_path)
if img_path.is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif img_path.is_dir():
    for f in glob.glob(args.img_path+'/*'):
        # print(os.path.splitext(os.path.basename(f))[0])
        save_outputs(f, os.path.splitext(os.path.basename(f))[0])
        # exit()
else:
    print("invalid file path!")
    sys.exit()