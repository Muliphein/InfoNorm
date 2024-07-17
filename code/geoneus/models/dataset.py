import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from PIL import Image
import matplotlib.pyplot as plt
import re
import open3d as o3d
import struct
import collections
import pickle
from models.read_write_model import *
from tqdm import tqdm
# from plyfile import PlyData, PlyElement


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



# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # not R but R^-1
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.use_jaco = conf.get_bool('use_jaco', default=False)
        self.dino_min = conf.get_float('dino_min', default=-1.0)
        self.normal_min = conf.get_float('normal_min', default=-1.0)
        self.split_name = conf.get_string('split_name', default='Not Find Split')
        print(f'Use Jaco {self.use_jaco}, dinomin: {self.dino_min}, normalmin:{self.normal_min}, split_name : {self.split_name}')

        self.data_dir = conf.get_string('data_dir')
        self.render_cameras_name = conf.get_string('render_cameras_name')
        self.object_cameras_name = conf.get_string('object_cameras_name')

        num_views = conf.get_string('n_views', default='max')
        self.generator = torch.Generator(device='cuda')
        self.generator.manual_seed(np.random.randint(1e9))

        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)

        with open(os.path.join(self.data_dir, self.split_name), 'rb') as file:
            serialized_list = file.read()
            train_list, eval_list = pickle.loads(serialized_list)

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = [os.path.join(self.data_dir, 'image', name) for name in train_list]
        self.n_images = len(self.images_lis)
        print(f'n Images : {self.n_images}')
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.images_gray_np = np.stack([cv.imread(im_name, cv.IMREAD_GRAYSCALE) for im_name in self.images_lis]) / 255.0  # Read grayscale images
        self.masks_np = np.ones_like(self.images_np[:, :, :, 0:1])

        self.world_mats_np = [camera_dict['world_mat_%s' % os.path.basename(name).split('.')[0]].astype(np.float32) for name in train_list]
        self.scale_mats_np = [camera_dict['scale_mat_%s' % os.path.basename(name).split('.')[0]].astype(np.float32) for name in train_list]
        
        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cuda()  # [n_images, H, W, 3]
        self.images_gray = torch.from_numpy(self.images_gray_np.astype(np.float32)).cuda()
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cuda()   # [n_images, H, W, 3]
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        pts_dir = os.path.join(self.data_dir, 'points.pkl')
        self.view_data = []
        with open(pts_dir, 'rb') as f:
            original_view_data = pickle.load(f)
            for item in original_view_data:
                if item.shape[0] == 0:
                    self.view_data.append(None)
                else:
                    item = np.concatenate([item, np.ones_like(item[:, 0:1])], axis=-1)
                    # print(item.shape)
                    item = item.transpose()
                    # print(item.shape)
                    scale_mat = np.linalg.inv(self.scale_mats_np[0])
                    item = scale_mat @ item
                    item = (item[:3, :]/item[3:4, :]).transpose()
                    # print(item.shape, ' ', item.min(), '<->', item.max())
                    # print(scale_mat.shape)
                    self.view_data.append(torch.from_numpy(item).float().cuda())

        if self.use_jaco:
            self.normals = []
            for item in tqdm(train_list, desc='Normal'):
                normals_npz = np.load(os.path.join(self.data_dir, 'pred_normal', item[:-4]+'.npz'))['arr_0']

                # no trans due to no use , no transform to world coordinates
                self.normals.append(normals_npz.astype(np.float32).reshape(self.H, self.W, 3))

            self.normals = np.stack(self.normals)
            self.normals = torch.from_numpy(self.normals).cuda()
            print('Normals Torch ', self.normals.shape)
            
            self.dinos = []
            self.dino_torch = torch.zeros((len(self.normals), self.H, self.W, 384), device='cpu')
            cnt = 0
            for item in tqdm(train_list, desc='Dino'):
                dino_single_torch = torch.load(os.path.join(self.data_dir, 'dino', item[:-4]+'.pth')).cpu().reshape(89, 119, 384)
                dino_single_np = dino_single_torch.cpu().numpy()
                # dino_single_resize = cv.resize(dino_single_np, (self.W, self.H), interpolation=cv.INTER_AREA)
                dino_single_resize = vit_feature_resize(dino_single_np, half_kernel_size=4, stride=4)
                self.dino_torch[cnt] = torch.from_numpy(dino_single_resize).cpu()
                del dino_single_torch
                del dino_single_resize
                del dino_single_np
                cnt = cnt+1
            print('Dino Torch ', self.dino_torch.shape)

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        # Object scale mat: region of interest to **extract mesh**
        object_scale_mat = np.load(os.path.join(self.data_dir, self.object_cameras_name))[f'scale_mat_{train_list[0][:-4]}']
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        self.num_views = self.n_images

        print(f'Total Views : {self.num_views}')
        self.src_idx = []
        for p in range(self.num_views):
            temp = []
            for i in range(max(0, p-5), min(self.num_views, p+5)):
                if i != p:
                    temp.append(i)
            self.src_idx.append(torch.tensor(temp))
        
        print(f'Srcidx : {len(self.src_idx)}')

        print('Load data: End')
        # exit()

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """

        src_idx = self.src_idx[img_idx]
        src_idx = src_idx[:9]
        idx_list = torch.cat([torch.tensor(img_idx).unsqueeze(0), src_idx], dim=0)

        poses_pair = self.pose_all[idx_list]  # [store R^-1 and C]
        intrinsics_pair = self.intrinsics_all[idx_list]
        intrinsics_inv_pair = self.intrinsics_all_inv[idx_list]
        images_gray_pair = self.images_gray[idx_list]

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)

        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1), intrinsics_pair, intrinsics_inv_pair, poses_pair, images_gray_pair

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """

        src_idx = self.src_idx[img_idx]
        src_idx = src_idx[:9]
        # src_idx = src_idx[-9:]
        idx_list = torch.cat([img_idx.clone().detach().unsqueeze(0), src_idx], dim=0).cuda()

        poses_pair = self.pose_all[idx_list]  # [store R^-1 and C]
        intrinsics_pair = self.intrinsics_all[idx_list]
        intrinsics_inv_pair = self.intrinsics_all_inv[idx_list]
        images_gray_pair = self.images_gray[idx_list]

        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        # print(pixels_x.device, pixels_y.device, self.images.device)
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        # print(pixels_x.device, pixels_y.device, self.masks.device)
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
        if self.use_jaco:
            dino_info = self.dino_torch[img_idx][(pixels_y.cpu(), pixels_x.cpu())].cuda()
            dino_info_norm = dino_info / (torch.linalg.norm(dino_info, dim=-1).unsqueeze(-1) + 1e-7)
            dino_info_cos = dino_info_norm @ torch.transpose(dino_info_norm, 0, 1)
            dino_grid = dino_info_cos > self.dino_min # [bs, bs]

            normal_info = self.normals[img_idx][(pixels_y, pixels_x)].cuda()
            normal_info_norm = normal_info / (torch.linalg.norm(normal_info, dim=-1).unsqueeze(-1) + 1e-7)
            normal_info_cos = normal_info_norm @ torch.transpose(normal_info_norm, 0, 1)
            normal_grid = normal_info_cos > self.normal_min # [bs, bs]

            jaco_info = dino_grid & normal_grid # [bs, bs]
        else:
            jaco_info = None

        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1).cuda(), intrinsics_pair, intrinsics_inv_pair, poses_pair, images_gray_pair, jaco_info    # batch_size, 10

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        # torch
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near.clip(5e-2, 1e9), far


    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def gen_pts_view(self, img_idx):
        return self.view_data[img_idx]
