'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import json
import numpy as np
import torch
import os
import torchvision.transforms.functional as torchvision_F
from PIL import Image, ImageFile
from tqdm import tqdm
import pickle
import copy
import cv2 as cv

from projects.nerf.datasets import base
from projects.nerf.utils import camera

ImageFile.LOAD_TRUNCATED_IMAGES = True

def vit_feature_resize(feature_map, half_kernel_size=4, stride = 4):
    # feature map to Pixel Info
    # Note that ViT are predict in block
    # we need a different upsample method to generate pixel predict
    # Half Kernelsize | Stride | ...
    # --------------- F ------ F ...
    H, W, C = feature_map.shape
    hafl_kernel_I = np.ones([half_kernel_size, half_kernel_size])

    pix_val = np.zeros([(H+1)*stride, (W+1)*stride, C])
    pix_sum = np.zeros([(H+1)*stride, (W+1)*stride, C])
    for dx in range(-stride, stride):
        for dy in range(-stride, stride):
            lx = abs(dx + 0.5)
            ly = abs(dy + 0.5)
            rate = (1 - lx / stride) * (1 - ly / stride)
            pix_val[stride+dx::stride, stride+dy::stride, :][:H, :W, :] = pix_val[stride+dx::stride, stride+dy::stride, :][:H, :W, :] + feature_map * rate
            pix_sum[stride+dx::stride, stride+dy::stride, :][:H, :W, :] = pix_sum[stride+dx::stride, stride+dy::stride, :][:H, :W, :] + rate
    pix_val = pix_val / pix_sum
    return pix_val


class Dataset(base.Dataset):

    def __init__(self, cfg, is_inference=False, is_test=False):
        super().__init__(cfg, is_inference=is_inference, is_test=False)
        cfg_data = cfg.data
        # print(cfg)
        self.root = cfg_data.root
        self.preload = cfg_data.preload
        self.dino_min = cfg_data.dino_min
        self.normal_min = cfg_data.normal_min
        self.H, self.W = cfg_data.val.image_size if is_inference else cfg_data.train.image_size
        print(f'Self.H W {self.H}x{self.W}')
        meta_fname = f"{cfg_data.root}/transforms.json"
        if is_test:
            meta_fname = f"{cfg_data.root}/transforms_novel.json"
        with open(meta_fname) as file:
            self.meta = json.load(file)
        self.list = self.meta["frames"]

        
        train_split_file = cfg_data.split
        pkl_file = os.path.join(cfg_data.root, train_split_file)
        
        with open(pkl_file, 'rb') as file:
            serialized_list = file.read()
            train_list, eval_list = pickle.loads(serialized_list)


        train_meta = []
        for item in self.list:
            if os.path.basename(item["file_path"]) in train_list:
                train_meta.append(item)
        self.list = train_meta
        print(f'Total File = {len(self.list)}')
        
        if cfg_data[self.split].subset:
            subset = cfg_data[self.split].subset
            subset_idx = np.linspace(0, len(self.list), subset+1)[:-1].astype(int)
            self.list = [self.list[i] for i in subset_idx]

        self.num_rays = cfg.model.render.rand_rays
        self.readjust = getattr(cfg_data, "readjust", None)
        
        self.use_jacobi = cfg.use_jacobi

        # only train are obtained
        self.name_list = [os.path.basename(self.list[i]["file_path"]) for i in range(len(self.list))]
        print(f'Pic List : {self.name_list}')

        assert self.preload, "Must Preload"
        if self.preload:
            
            self.images = []

            for item in tqdm(self.list, desc='Images'):
                fpath = item["file_path"]
                image_fname = os.path.join(self.root, fpath)
                image = Image.open(image_fname)
                self.image_size_raw = image.size
                image = image.resize((self.W, self.H))
                image.load()
                self.images.append(image)

            self.preprocess_image_all()
            self.images = torch.stack(self.images).cuda()
            print('Images shape : ', self.images.shape)
            self.cameras = []
            for item in tqdm(self.list, desc='Cameras'):
                intr = torch.tensor([[self.meta["fl_x"], self.meta["sk_x"], self.meta["cx"]],
                                    [self.meta["sk_y"], self.meta["fl_y"], self.meta["cy"]],
                                    [0, 0, 1]]).float().cpu()
                # Camera pose.
                c2w_gl = torch.tensor(item["transform_matrix"], dtype=torch.float32).cpu()
                c2w = self._gl_to_cv(c2w_gl)
                # center scene
                center = np.array(self.meta["sphere_center"])
                center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
                c2w[:3, -1] -= center
                # scale scene
                scale = np.array(self.meta["sphere_radius"])
                scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
                c2w[:3, -1] /= scale
                w2c = camera.Pose().invert(c2w[:3])

                intr = intr.clone()
                raw_W, raw_H = self.image_size_raw
                intr[0] *= self.W / raw_W
                intr[1] *= self.H / raw_H

                self.cameras.append((intr, w2c))

            if cfg.use_jacobi and self.split == "train":
                self.normals = []
                for item in tqdm(self.list, desc='Normal'):
                    base_name = os.path.basename(item["file_path"])
                    normals_npz = np.load(os.path.join(cfg_data.root, 'pred_normal', base_name[:-4]+'.npz'))['arr_0']
                    # no trans due to no use , no transform to world coordinates
                    self.normals.append(normals_npz.astype(np.float32).reshape(self.H, self.W, 3))

                self.normals = np.stack(self.normals)
                self.normals = torch.from_numpy(self.normals).cuda()
                print('Normals Torch ', self.normals.shape)
                
                self.dinos = []
                self.dino_torch = torch.zeros((len(self.normals), self.H, self.W, 384), device='cpu')
                cnt = 0
                for item in tqdm(self.list, desc='Dino'):
                    base_name = os.path.basename(item["file_path"])
                    dino_single_torch = torch.load(os.path.join(cfg_data.root, 'dino', base_name[:-4]+'.pth')).cpu().reshape(89, 119, 384)
                    dino_single_np = dino_single_torch.cpu().numpy()
                    # dino_single_resize = cv.resize(dino_single_np, (self.W, self.H), interpolation=cv.INTER_AREA)
                    dino_single_resize = vit_feature_resize(dino_single_np, half_kernel_size=4, stride=4)
                    self.dino_torch[cnt] = torch.from_numpy(dino_single_resize).cpu()
                    del dino_single_torch
                    del dino_single_resize
                    del dino_single_np
                    cnt = cnt+1
                print('Dino Torch ', self.dino_torch.shape)

    def preprocess_image_all(self,):
        images = []
        for image in self.images:
            image = self.preprocess_image(image)
            images.append(image)
        
        self.images = images

    def get_subset(self):
        return self.subset_idx

    def __getitem__(self, idx):
        """Process raw data and return processed data in a dictionary.

        Args:
            idx: The index of the sample of the dataset.
        Returns: A dictionary containing the data.
                 idx (scalar): The index of the sample of the dataset.
                 image (R tensor): Image idx for per-image embedding.
                 image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
                 intr (3x3 tensor): The camera intrinsics of `image`.
                 pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
        """
        # Keep track of sample index for convenience.
        sample = dict(idx=idx)
        # Get the images.
        # print(f'Idx = {idx}')
        image = self.images[idx]

        intr, pose = self.cameras[idx]
        
        if self.use_jacobi and self.split == "train":
            normal = self.normals[idx]
            dino = self.dino_torch[idx]
        
        # Pre-sample ray indices.
        if self.split == "train":
            ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
            image_sampled = image.flatten(1, 2)[:, ray_idx].t()  # [R,3]

            if self.use_jacobi and self.split == "train":
                normal_sampled = normal.flatten(0, 1)[ray_idx.cpu(), :]
                dino_sampled = dino.flatten(0, 1)[ray_idx.cpu(), :].cuda()
                # print('Normal Sampled : ', normal_sampled.shape)
                # print('Dino Sampled : ', dino_sampled.shape)
            
                dino_info_norm = dino_sampled / (torch.linalg.norm(dino_sampled, dim=-1).unsqueeze(-1) + 1e-7)
                dino_info_cos = dino_info_norm @ torch.transpose(dino_info_norm, 0, 1)
                dino_grid = dino_info_cos > self.dino_min # [bs, bs]

                normal_info_norm = normal_sampled / (torch.linalg.norm(normal_sampled, dim=-1).unsqueeze(-1) + 1e-7)
                normal_info_cos = normal_info_norm @ torch.transpose(normal_info_norm, 0, 1)
                normal_grid = normal_info_cos > self.normal_min # [bs, bs]

                jaco_info = dino_grid & normal_grid # [bs, bs]

                sample.update(
                    ray_idx=ray_idx,
                    image_sampled=image_sampled,
                    intr=intr.cuda(),
                    pose=pose.cuda(),
                    jaco_info=jaco_info,
                )
            else:
                sample.update(
                    ray_idx=ray_idx,
                    image_sampled=image_sampled,
                    intr=intr.cuda(),
                    pose=pose.cuda(),
                )

        else:  # keep image during inference
            sample.update(
                image=image,
                intr=intr,
                pose=pose,
            )
        return sample

    def preprocess_image(self, image):
        image = image.resize((self.W, self.H))
        image = torchvision_F.to_tensor(image)
        rgb = image[:3]
        return rgb

    def _gl_to_cv(self, gl):
        # convert to CV convention used in Imaginaire
        cv = gl * torch.tensor([1, -1, -1, 1]).cpu()
        return cv

# class Dataset(base.Dataset):

#     def __init__(self, cfg, is_inference=False, is_test=False):
#         super().__init__(cfg, is_inference=is_inference, is_test=False)
#         cfg_data = cfg.data
#         self.root = cfg_data.root
#         self.preload = cfg_data.preload
#         self.H, self.W = cfg_data.val.image_size if is_inference else cfg_data.train.image_size
#         meta_fname = f"{cfg_data.root}/transforms.json"
#         if is_test:
#             meta_fname = f"{cfg_data.root}/transforms_novel.json"
#         with open(meta_fname) as file:
#             self.meta = json.load(file)
#         self.list = self.meta["frames"]

#         if cfg_data[self.split].subset:
#             subset = cfg_data[self.split].subset
#             subset_idx = np.linspace(0, len(self.list), subset+1)[:-1].astype(int)
#             self.list = [self.list[i] for i in subset_idx]


#         self.name_list = [os.path.basename(self.list[i]["file_path"]) for i in range(len(self.list))]
#         print(f'Pic List : {self.name_list}')
#         self.num_rays = cfg.model.render.rand_rays
#         self.readjust = getattr(cfg_data, "readjust", None)
#         if cfg_data.preload:
#             self.images = self.preload_threading(self.get_image, 4)
#             self.cameras = self.preload_threading(self.get_camera, 4, data_str="cameras")

#     def get_subset(self):
#         return self.subset_idx

#     def __getitem__(self, idx):
#         """Process raw data and return processed data in a dictionary.

#         Args:
#             idx: The index of the sample of the dataset.
#         Returns: A dictionary containing the data.
#                  idx (scalar): The index of the sample of the dataset.
#                  image (R tensor): Image idx for per-image embedding.
#                  image (Rx3 tensor): Image with pixel values in [0,1] for supervision.
#                  intr (3x3 tensor): The camera intrinsics of `image`.
#                  pose (3x4 tensor): The camera extrinsics [R,t] of `image`.
#         """
#         # Keep track of sample index for convenience.
#         sample = dict(idx=idx)
#         image, image_size_raw = self.images[idx] if self.preload else self.get_image(idx)
        
#         image0, image_size_raw0 = self.images[0]
#         image0 = self.preprocess_image(image0)

#         image = self.preprocess_image(image)
#         intr, pose = self.cameras[idx] if self.preload else self.get_camera(idx)
#         intr, pose = self.preprocess_camera(intr, pose, image_size_raw)

#         intr0, pose0 = self.cameras[0]
#         intr0, pose0 = self.preprocess_camera(intr0, pose0, image_size_raw0)

#         if self.split == "train":
#             ray_idx = torch.randperm(self.H * self.W)[:self.num_rays]  # [R]
#             image_sampled = image.flatten(1, 2)[:, ray_idx.cpu()].t()  # [R,3]

#             print('Image Break : ', image0.flatten(1, 2)[:, 500:505].t())
#             print(f'intr : {intr0}, pose : {pose0}')

#             sample.update(
#                 ray_idx=ray_idx,
#                 image_sampled=image_sampled,
#                 intr=intr,
#                 pose=pose,
#             )
#         else:  # keep image during inference
#             sample.update(
#                 image=image,
#                 intr=intr,
#                 pose=pose,
#             )
#         return sample

#     def get_image(self, idx):
#         fpath = self.list[idx]["file_path"]
#         image_fname = f"{self.root}/{fpath}"
#         image = Image.open(image_fname)
#         image.load()
#         image_size_raw = image.size
#         return image, image_size_raw

#     def preprocess_image(self, image):
#         # Resize the image.
#         image = image.resize((self.W, self.H))
#         image = torchvision_F.to_tensor(image)
#         rgb = image[:3]
#         return rgb

#     def get_camera(self, idx):
#         # Camera intrinsics.
#         intr = torch.tensor([[self.meta["fl_x"], self.meta["sk_x"], self.meta["cx"]],
#                              [self.meta["sk_y"], self.meta["fl_y"], self.meta["cy"]],
#                              [0, 0, 1]]).float()
#         # Camera pose.
#         c2w_gl = torch.tensor(self.list[idx]["transform_matrix"], dtype=torch.float32)
#         c2w = self._gl_to_cv(c2w_gl)
#         # center scene
#         center = np.array(self.meta["sphere_center"])
#         center += np.array(getattr(self.readjust, "center", [0])) if self.readjust else 0.
#         c2w[:3, -1] -= center
#         # scale scene
#         scale = np.array(self.meta["sphere_radius"])
#         scale *= getattr(self.readjust, "scale", 1.) if self.readjust else 1.
#         c2w[:3, -1] /= scale
#         w2c = camera.Pose().invert(c2w[:3])
#         return intr, w2c

#     def preprocess_camera(self, intr, pose, image_size_raw):
#         # Adjust the intrinsics according to the resized image.
#         intr = intr.clone()
#         raw_W, raw_H = image_size_raw
#         intr[0] *= self.W / raw_W
#         intr[1] *= self.H / raw_H
#         return intr, pose

#     def _gl_to_cv(self, gl):
#         # convert to CV convention used in Imaginaire
#         cv = gl * torch.tensor([1, -1, -1, 1])
#         return cv
