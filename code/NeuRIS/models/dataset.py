import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import cv2 as cv
import numpy as np
import os, copy, logging

from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from utils.utils_image import read_images, write_image, write_images
from utils.utils_io import checkExistence, ensure_dir_existence, get_path_components, get_files_stem
from utils.utils_geometry import get_pose_inv, get_world_normal, quat_to_rot, save_points

import utils.utils_geometry as GeoUtils
import utils.utils_io as IOUtils
import models.patch_match_cuda as PatchMatch
import utils.utils_training as TrainingUtils
import utils.utils_image as ImageUtils

import pickle


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
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class Dataset:
    '''Check normal and depth in folder depth_cloud
    '''
    def __init__(self, conf, mode='train'):
        super(Dataset, self).__init__()
        # logging.info('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.load_dino = conf.get_bool('load_dino', default=False)
        self.jaco_info = conf.get_string('jaco_info', default='jaco')
        self.jacobi_start = conf.get_int('jacobi_start', default=0)
        self.jacobi_end = conf.get_int('jacobi_end', default=1e-9)
        self.near_plane = conf.get_float('near_plane', default=5e-2)
        self.normal_perturb = conf.get_string('normal_perturb', default='')
        self.semantic_type = conf.get_string('semantic_type', default='dino')
        print('Semantic Type, ',  self.semantic_type)
        self.jacobi_pos_threshold = conf.get_float('jacobi_pos_threshold', default=None)
        if self.jaco_info == 'combine_double_min':
            self.dino_min = conf.get_float('dino_min', default=0.6)
            self.normal_min = conf.get_float('normal_min', default=0.985)
            print(f'Dino Low : {self.dino_min}, Normal Min : {self.normal_min}')
        else:
            assert self.jacobi_pos_threshold!=None
        
        print(f'Near Plane = {self.near_plane}')
        print(f'Jaco Info From {self.jaco_info}')
        self.data_dir = conf['data_dir']
        print('Normal Folder = ', os.path.join(self.data_dir, 'pred_normal'+self.normal_perturb))
        self.piece_size = 2**20
        self.bbox_size_half = conf['bbox_size_half']
        self.use_normal = True
        self.split_name = conf['split_name']

        self.denoise_gray_image = self.conf['denoise_gray_image']
        self.denoise_paras = self.conf['denoise_paras']

        self.use_planes = conf['use_planes']
        self.use_plane_offset_loss = conf['use_plane_offset_loss']
 
        self.cameras_name = './cameras_sphere.npz'
        path_cam = os.path.join(self.data_dir, './cameras_sphere.npz')  # cameras_sphere, cameras_linear_init
        camera_dict = np.load(path_cam)
        logging.info(f'Load camera dict: {path_cam.split("/")[-1]}')
        
        with open(os.path.join(self.data_dir, self.split_name), 'rb') as file:
            serialized_list = file.read()
            train_list, eval_list = pickle.loads(serialized_list)

        self.images_lis = [os.path.join(self.data_dir, 'image', i) for i in train_list]
        self.names_lis = [os.path.basename(name).split('.')[0] for name in self.images_lis]
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([self.read_img(im_name) for im_name in self.images_lis]) / 256.0
        self.masks_np = np.ones_like(self.images_np)[:, :, :, 0]
        
        self.eval_images_lis = [os.path.join(self.data_dir, 'image', i) for i in eval_list]
        self.eval_names_lis = [os.path.basename(name).split('.')[0] for name in self.eval_images_lis]

        print('Images Np ', self.images_np.shape)
        print('Masks Np ', self.masks_np.shape)

        self.world_mats_np = [camera_dict['world_mat_%s' % name].astype(np.float32) for name in self.names_lis]
        self.scale_mats_np = [camera_dict['scale_mat_%s' % name].astype(np.float32) for name in self.names_lis]
        
        self.eval_world_mats_np = [camera_dict['world_mat_%s' % name].astype(np.float32) for name in self.eval_names_lis]
        self.eval_scale_mats_np = [camera_dict['scale_mat_%s' % name].astype(np.float32) for name in self.eval_names_lis]

        self.intrinsics_all = []
        self.pose_all = []

        # i = 0
        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.eval_intrinsics_all = []
        self.eval_pose_all = []

        for scale_mat, world_mat in zip(self.eval_scale_mats_np, self.eval_world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = load_K_Rt_from_P(None, P)
            self.eval_intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.eval_pose_all.append(torch.from_numpy(pose).float())

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cuda()  # n_images, H, W, 3   # Save GPU memory
        self.masks  = torch.from_numpy(self.masks_np.astype(np.float32)).cuda()   # n_images, H, W   # Save GPU memory
        h_img, w_img, c_img = self.images[0].shape
        print(f"Image size: ({w_img}, {h_img}, {c_img})")

        if self.load_dino and mode == 'train':
            if self.semantic_type == 'dino':
                self.dino_lis = [os.path.join(self.data_dir, 'dino', i[:-4] + '.pth') for i in train_list]
                print(self.dino_lis)
                
                #? https://medium.com/@wenrudong/what-is-opencvs-inter-area-actually-doing-282a626a09b3
                #? How cv.INTER_AREA work
                #? It seems that the cv and pillow not equal

                dino_load_bar = tqdm(total=len(self.dino_lis))
                self.dino_torch = torch.zeros((len(self.dino_lis), h_img, w_img, 384), device='cpu')
                for idx, dino_name in enumerate(self.dino_lis):
                    dino_single_torch = torch.load(dino_name).cpu().reshape(89, 119, 384)
                    dino_single_np = dino_single_torch.cpu().numpy()
                    dino_single_resize = cv.resize(dino_single_np, (w_img, h_img), interpolation=cv.INTER_AREA)
                    # dino_single_resize = vit_feature_resize(dino_single_np, half_kernel_size=4, stride=4)
                    self.dino_torch[idx] = torch.from_numpy(dino_single_resize).cpu()
                    dino_load_bar.update(1)
                    del dino_single_torch
                    del dino_single_resize
                    del dino_single_np

                print('Dino Torch ', self.dino_torch.shape)
            elif self.semantic_type == 'sam':
                self.dino_lis = [os.path.join(self.data_dir, 'sam', i) for i in train_list]
                print(self.dino_lis)
                dino_load_bar = tqdm(total=len(self.dino_lis))
                self.dino_torch = torch.zeros((len(self.dino_lis), h_img, w_img, 3), device='cpu')
                for idx, dino_name in enumerate(self.dino_lis):
                    dino_single_torch = cv.imread(dino_name) / 256.0
                    self.dino_torch[idx] = torch.from_numpy(dino_single_torch).cpu()
                    dino_load_bar.update(1)
                    del dino_single_torch

                print('Dino Torch ', self.dino_torch.shape)

            else:
                raise NotImplementedError


        if self.use_normal:
            print(f'[Use normal] Loading estimated normals...')
            normals_np = []
            for i in tqdm(range(self.n_images)):
                normals_npz = np.load(os.path.join(self.data_dir, 'pred_normal'+self.normal_perturb, self.names_lis[i]+'.npz'))['arr_0']
                normal_img_curr = normals_npz
                # transform to world coordinates
                ex_i = torch.linalg.inv(self.pose_all[i])
                img_normal_w = get_world_normal(normal_img_curr.reshape(-1, 3), ex_i).reshape(h_img, w_img,3)
                normals_np.append(img_normal_w)
            self.normals_np = -np.stack(normals_np)   # reverse normal
            self.normals = torch.from_numpy(self.normals_np.astype(np.float32)).cuda()

        
            print('Normals Np ', self.normals_np.shape)

        if self.use_planes:
            assert False, "Use Planes?"

        if self.use_plane_offset_loss:
            assert False, "Use Plane Offset?"

        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)  # n, 4, 4
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all) # n, 4, 4

        self.eval_intrinsics_all = torch.stack(self.eval_intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.eval_intrinsics_all_inv = torch.inverse(self.eval_intrinsics_all)  # [n_images, 4, 4]

        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # n_images, 4, 4
        self.eval_pose_all = torch.stack(self.eval_pose_all).to(self.device)  # n_images, 4, 4
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        # for patch match
        self.min_neighbors_ncc = 3
        print('No Neighbors!')

        self.initialize_patchmatch()
        
        # Gen train_data
        self.train_data = None
        self.sphere_radius =  conf['sphere_radius']
        self.bbox_min = np.array([-1.01*self.bbox_size_half, -1.01*self.bbox_size_half, -1.01*self.bbox_size_half])
        self.bbox_max = np.array([ 1.01*self.bbox_size_half,  1.01*self.bbox_size_half,  1.01*self.bbox_size_half])

        self.iter_step = 0
        
    def initialize_patchmatch(self):
        self.check_occlusion = self.conf['check_occlusion']
        
        logging.info(f'Prepare gray images...')
        self.extrinsics_all = torch.linalg.inv(self.pose_all)
        self.images_gray = []
        self.images_denoise_np = []

        if self.denoise_gray_image:
            dir_denoise = os.path.join(self.data_dir, 'image_denoised')
            if not checkExistence(dir_denoise) and len(get_files_stem(dir_denoise, '.png'))==0:
                assert('Denoised Failed')
            else:
                logging.info(f'Load predenoised images by openCV structural denoise: {dir_denoise}')
                self.images_denoise_np = np.stack([self.read_img(os.path.join(dir_denoise, im_name+'.png')) for im_name in self.names_lis])
        else:
            logging.info(f'Use original image to generate gray image...')
            self.images_denoise_np = self.images_np * 255

        for i in tqdm(range(self.n_images)):
            img_gray = cv.cvtColor(self.images_denoise_np[i].astype(np.uint8), cv.COLOR_BGR2GRAY)
            self.images_gray.append(img_gray)

        # range: (0,255)
        self.images_gray_np = np.array(self.images_gray).astype(np.float32)
        self.images_gray = torch.from_numpy(self.images_gray_np).cuda()

        # For cache rendered depths and normals
        self.confidence_accum = None
        self.samples_accum = None
        self.normals_accum = None
        self.depths_accum = None
        self.points_accum = None
        self.render_difference_accum = None
        self.samples_accum = torch.zeros_like(self.masks, dtype=torch.int32).cuda()
        
        b_accum_all_data = False
        if b_accum_all_data:
            self.colors_accum = torch.zeros_like(self.images, dtype=torch.float32).cuda()
                    
    def read_img(self, path):
        img = cv.imread(path)
        H, W = img.shape[0], img.shape[1]

        return img

    def estimated_scale_mat(self):
        assert len(self.world_mats_np) > 0
        rays_o = []
        rays_v = []
        for world_mat in self.world_mats_np:
            P = world_mat[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            rays_o.append(c2w[:3, 3])
            rays_v.append(c2w[:3, 0])
            rays_o.append(c2w[:3, 3])
            rays_v.append(c2w[:3, 1])

        rays_o = np.stack(rays_o, axis=0)   # N * 3
        rays_v = np.stack(rays_v, axis=0)   # N * 3
        dot_val = np.sum(rays_o * rays_v, axis=-1, keepdims=True)  # N * 1
        center, _, _, _ = np.linalg.lstsq(rays_v, dot_val)
        center = center.squeeze()
        radius = np.max(np.sqrt(np.sum((rays_o - center[None, :])**2, axis=-1)))
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center
        scale_mat = scale_mat.astype(np.float32)
        scale_mats = [scale_mat for _ in self.world_mats_np]

        return scale_mats

    def gen_rays(self):
        tx = torch.linspace(0, self.W - 1, self.W)
        ty = torch.linspace(0, self.H - 1, self.H)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[:, None, None, :3, :3], p[None, :, :, :, None]).squeeze() # n, W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # n, W, H, 3
        rays_v = torch.matmul(self.pose_all[:, None, None, :3, :3],  rays_v[:, :, :, :, None]).squeeze()  # n, W, H, 3
        rays_o = self.pose_all[:, None, None, :3, 3].expand(rays_v.shape)  # n, W, H, 3
        return rays_o.transpose(1, 2), rays_v.transpose(1, 2)

    def get_pose(self, img_idx, pose):
        pose_cur = None
        if pose == None:
            pose_cur = self.pose_all[img_idx]
        elif pose is not None:
            if pose.dim() == 1:
                pose = pose.unsqueeze(0)
            assert pose.dim() == 2
            if pose.shape[1] == 7: #In case of quaternion vector representation
                cam_loc = pose[:, 4:]
                R = quat_to_rot(pose[:,:4])
                p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
                p[:, :3, :3] = R
                p[:, :3, 3] = cam_loc
            else: # In case of pose matrix representation
                cam_loc = pose[:, :3, 3]
                p = pose
            pose_cur = p
        else:
            NotImplementedError 

        return pose_cur.squeeze()

    def gen_rays_at_pose(self, pose, intr_inv, resolution_level=1):
        """
        Generate rays at world space from one camera (c2w).
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l).cuda()
        ty = torch.linspace(0, self.H - 1, self.H // l).cuda()
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float().cuda() # W, H, 3
        p = torch.matmul(intr_inv[None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_at(self, img_idx, pose = None, resolution_level=1):
        pose_cur = self.get_pose(img_idx, pose)

        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(pose_cur[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = pose_cur[None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
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
        key_rots = [rot_0, rot_1]
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

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)][:, None]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
        if self.iter_step >= self.jacobi_start and self.iter_step < self.jacobi_end:
            if self.jaco_info == 'dino':
                jaco_info = self.dino_torch[img_idx][(pixels_y.cpu(), pixels_x.cpu())].cuda() if self.load_dino else None
            elif self.jaco_info == 'normal':
                jaco_info = self.normals[img_idx][(pixels_y, pixels_x)].cuda()
            elif self.jaco_info == 'combine_double_min':
                raise NotImplementedError(f'No Such Jacoinfo {self.jaco_info}')
            else:
                raise NotImplementedError(f'No Such Jacoinfo {self.jaco_info}')
        else:
            jaco_info = None
        return torch.cat([rays_o, rays_v, color, mask[:, :1]], dim=-1).cuda(), jaco_info    # batch_size, 10
        
    def random_get_rays_at(self, img_idx, batch_size, pose = None):
        pose_cur = self.get_pose(img_idx, pose)
        
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
            
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)][:, None]     # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(pose_cur[None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = pose_cur[None, :3, 3].expand(rays_v.shape) # batch_size, 3
        
        normal_sample = None
        if self.use_normal:
            normal_sample = self.normals[img_idx][(pixels_y, pixels_x)].cuda()

        planes_sample = None
        if self.use_planes:
            planes_sample = self.planes[img_idx][(pixels_y, pixels_x)].unsqueeze(-1).cuda()
        
        subplanes_sample = None
        if self.use_plane_offset_loss:
            subplanes_sample = self.subplanes[img_idx][(pixels_y, pixels_x)].unsqueeze(-1).cuda()
        dino_mask = None
        if self.iter_step >= self.jacobi_start and self.iter_step < self.jacobi_end:
            if self.jaco_info == 'dino':
                if not self.load_dino:
                    jaco_info = None
                else:
                    dino_info = self.dino_torch[img_idx][(pixels_y.cpu(), pixels_x.cpu())].cuda()
                    dino_info_norm = dino_info / (torch.linalg.norm(dino_info, dim=-1).unsqueeze(-1) + 1e-7)
                    dino_info_cos = dino_info_norm @ torch.transpose(dino_info_norm, 0, 1)
                    jaco_info = dino_info_cos > self.jacobi_pos_threshold # [bs, bs]
            elif self.jaco_info == 'normal':
                normal_info = self.normals[img_idx][(pixels_y, pixels_x)].cuda()
                normal_info_norm = normal_info / (torch.linalg.norm(normal_info, dim=-1).unsqueeze(-1) + 1e-7)
                normal_info_cos = normal_info_norm @ torch.transpose(normal_info_norm, 0, 1)
                jaco_info = normal_info_cos > self.jacobi_pos_threshold # [bs, bs]
            elif self.jaco_info == 'combine_double_min':
                
                dino_info = self.dino_torch[img_idx][(pixels_y.cpu(), pixels_x.cpu())].cuda()
                dino_info_norm = dino_info / (torch.linalg.norm(dino_info, dim=-1).unsqueeze(-1) + 1e-7)
                dino_mask = ~torch.all(dino_info == 0, dim=1)
                dino_info_cos = dino_info_norm @ torch.transpose(dino_info_norm, 0, 1)
                dino_grid = dino_info_cos > self.dino_min # [bs, bs]

                normal_info = self.normals[img_idx][(pixels_y, pixels_x)].cuda()
                normal_info_norm = normal_info / (torch.linalg.norm(normal_info, dim=-1).unsqueeze(-1) + 1e-7)
                normal_info_cos = normal_info_norm @ torch.transpose(normal_info_norm, 0, 1)
                normal_grid = normal_info_cos > self.normal_min # [bs, bs]

                jaco_info = dino_grid & normal_grid # [bs, bs]

            else:
                raise NotImplementedError(f'No Such Jacoinfo {self.jaco_info}')
        else:
            jaco_info = None

        return torch.cat([rays_o, rays_v, color, mask], dim=-1).cuda(), pixels_x, pixels_y, normal_sample, planes_sample, subplanes_sample, jaco_info, dino_mask    # batch_size, 10

    def near_far_from_sphere(self, rays_o, rays_d):
        # torch
        assert self.sphere_radius is not None
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        c = torch.sum(rays_o ** 2, dim=-1, keepdim=True) - self.sphere_radius**2
        mid = 0.5 * (-b) / a
        near = mid - self.sphere_radius
        far = mid + self.sphere_radius
        return near.clip(self.near_plane, 1e9), far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)

    def shuffle(self):
        r = torch.randperm(len(self.train_data))
        self.train_data = self.train_data[r]
        self.large_counter = 0
        self.small_counter = 0

    def next_train_batch(self, batch_size):
        if self.train_piece == None or self.small_counter + batch_size >= len(self.train_piece):
            if self.train_piece == None or self.large_counter + self.piece_size >= len(self.train_data):
                self.shuffle()
            self.train_piece_np = self.train_data[self.large_counter: self.large_counter + self.piece_size]
            self.train_piece = self.train_piece_np.cuda()
            self.small_counter = 0
            self.large_counter += self.piece_size

        curr_train_data = self.train_piece[self.small_counter: self.small_counter + batch_size]
        curr_train_data_np = self.train_piece_np[self.small_counter: self.small_counter + batch_size]
        self.small_counter += batch_size

        return curr_train_data, curr_train_data_np


    def score_pixels_ncc(self, idx, pts_world, normals_world, pixels_coords_vu, reso_level = 1.0, _debug = False):
        '''Use patch-match to evaluate the geometry: Smaller, better
        Return:
            scores_all_mean: N*1
            diff_patch_all: N*1
            mask_valid_all: N*1
        '''
        K = copy.deepcopy(self.intrinsics_all[0][:3,:3])
        img_ref = self.images_gray[idx]
        H, W = img_ref.shape
        window_size, window_step= 11, 2
        if reso_level > 1:
            K[:2,:3] /= reso_level
            img_ref = self.images_gray_np[idx]
            img_ref = cv.resize(img_ref, (int(W/reso_level), int(H/reso_level)), interpolation=cv.INTER_LINEAR)
            img_ref = torch.from_numpy(img_ref).cuda()
            window_size, window_step= (5, 1) if reso_level== 2 else (3, 1)

        if hasattr(self, 'dict_neighbors'):
            idx_neighbors = self.dict_neighbors[int(idx)]
            if len(idx_neighbors) < self.min_neighbors_ncc:
                return torch.ones(pts_world.shape[0]), torch.zeros(pts_world.shape[0]), torch.zeros(pts_world.shape[0]).bool()
        else:
            idx_neighbors = [idx-3, idx-2, idx-1, idx+1, idx+2, idx+3]
            if idx < 3:
                idx_neighbors = [idx+1, idx+2, idx+3]
            if idx > self.n_images-4:
                idx_neighbors = [idx-3, idx-2, idx-1]

        assert pixels_coords_vu.ndim == 2
        num_patches = pixels_coords_vu.shape[0]

        extrin_ref = self.extrinsics_all[idx]
        pts_ref = (extrin_ref[None,...] @ TrainingUtils.convert_to_homo(pts_world)[..., None]).squeeze()[:,:3]
        normals_ref = (extrin_ref[:3,:3][None,...] @ normals_world[..., None]).squeeze()
        
        patches_ref, idx_patch_pixels_ref, mask_idx_inside = PatchMatch.prepare_patches_src(img_ref, pixels_coords_vu, window_size, window_step)
        scores_all_mean, diff_patch_all, count_valid_all = torch.zeros(num_patches, dtype=torch.float32), torch.zeros(num_patches, dtype=torch.float32), torch.zeros(num_patches, dtype=torch.uint8)
        for idx_src in idx_neighbors:
            img_src = self.images_gray[idx_src]
            if reso_level > 1:
                img_src = cv.resize(self.images_gray_np[idx_src], (int(W/reso_level), int(H/reso_level)), interpolation=cv.INTER_LINEAR)
                img_src = torch.from_numpy(img_src).cuda()

            extrin_src = self.extrinsics_all[idx_src]

            homography = PatchMatch.compute_homography(pts_ref, normals_ref, K, extrin_ref, extrin_src)
            idx_patch_pixels_src = PatchMatch.warp_patches(idx_patch_pixels_ref, homography)
            patches_src = PatchMatch.sample_patches(img_src, idx_patch_pixels_src, sampling_mode = 'grid_sample')
            scores_curr, diff_patch_mean_curr, mask_patches_valid_curr = PatchMatch.compute_NCC_score(patches_ref, patches_src)

            # check occlusion
            if self.check_occlusion:
                mask_no_occlusion = scores_curr < 0.66
                mask_patches_valid_curr = mask_patches_valid_curr & mask_no_occlusion
                scores_curr[mask_no_occlusion==False] = 0.0
                diff_patch_mean_curr[mask_no_occlusion==False] = 0.0

            scores_all_mean += scores_curr
            diff_patch_all += diff_patch_mean_curr
            count_valid_all += mask_patches_valid_curr

            if _debug:
                corords_src = idx_patch_pixels_src[:,3,3].cpu().numpy().astype(int)
                img_sample_ref = PatchMatch.visualize_sampled_pixels(self.images[idx].numpy()*255, pixels_coords_vu.cpu().numpy())
                img_sample_src = PatchMatch.visualize_sampled_pixels(self.images[idx_src].numpy()*255, corords_src)
                ImageUtils.write_image_lis(f'./test/ncc/{idx}_{idx_src}.png', [img_sample_ref, img_sample_src])

                # save patches
                ImageUtils.write_image_lis(f'./test/ncc/patches_{idx}.png',[ patches_ref[i].cpu().numpy() for i in range(len(patches_ref))], interval_img = 5 )
                ImageUtils.write_image_lis(f'./test/ncc/patches_{idx_src}.png',[ patches_ref[i].cpu().numpy() for i in range(len(patches_src))], interval_img = 5 )

        
        # get the average scores of all neighbor views
        mask_valid_all = count_valid_all>=self.min_neighbors_ncc
        scores_all_mean[mask_valid_all] /= count_valid_all[mask_valid_all]
        diff_patch_all[mask_valid_all]  /= count_valid_all[mask_valid_all]

        # set unvalid scores and diffs to zero
        scores_all_mean = scores_all_mean*mask_valid_all
        diff_patch_all  = diff_patch_all*mask_valid_all


        scores_all_mean[mask_valid_all==False] = 1.0 # average scores for pixels without patch.

        return scores_all_mean, diff_patch_all, mask_valid_all
            