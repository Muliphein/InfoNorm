import json
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import utils
from utils import rend_util
from tqdm.contrib import tzip, tenumerate
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import pickle
from tqdm import tqdm
import cv2 as cv


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

class ReconDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 scan_id=0,
                 use_mask=False,
                 use_depth=False,
                 use_normal=False,
                 use_bubble=False,
                 use_lightmask=False,
                 use_jaco=False,
                 is_hdr=False,
                 dino_min=1e-9,
                 normal_min=1e-9,
                 **kwargs
                 ):
        self.sampling_idx = slice(None)

        self.instance_dir = os.path.join(data_dir, str(scan_id))
        self.data_dir = self.instance_dir
        self.split_name = kwargs['split']

        assert os.path.exists(self.instance_dir), "Data directory is empty"
        print(f"[INFO] Loading data from {self.instance_dir}")

        image_dir = f'{self.instance_dir}/image' if not is_hdr else '{0}/hdr'.format(self.instance_dir)

        self.is_hdr = is_hdr
        if is_hdr:
            print("[INFO] Using HDR image")

        if self.split_name == 'all.pkl':
            image_paths = sorted(utils.glob_imgs(image_dir))
        else:
            with open(os.path.join(self.data_dir, self.split_name), 'rb') as file:
                serialized_list = file.read()
                train_list, eval_list = pickle.loads(serialized_list)

            image_paths = [os.path.join(image_dir, train_file) for train_file in train_list]

        self.n_images = len(image_paths)
        print(f'Total Images : {self.n_images}')

        self.cam_file = '{0}/cameras_normalize.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        if 'scale_mat_0' in camera_dict.keys():
            print('Type x')
            scale_mats = [camera_dict['scale_mat_%s' % int(os.path.basename(name).split('.')[0])].astype(np.float32) for name in image_paths]
            world_mats = [camera_dict['world_mat_%s' % int(os.path.basename(name).split('.')[0])].astype(np.float32) for name in image_paths]
        else:
            print('Type 00x')
            scale_mats = [camera_dict['scale_mat_%s' % os.path.basename(name).split('.')[0]].astype(np.float32) for name in image_paths]
            world_mats = [camera_dict['world_mat_%s' % os.path.basename(name).split('.')[0]].astype(np.float32) for name in image_paths]


        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        self.intrinsics_all = torch.stack(self.intrinsics_all, 0)
        self.pose_all = torch.stack(self.pose_all, 0)

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path, is_hdr=is_hdr)
            self.img_res = [rgb.shape[1], rgb.shape[2]]
            c_img, h_img, w_img = rgb.shape
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        self.rgb_images = torch.stack(self.rgb_images, 0)
        print(f'Image Tensor : {self.rgb_images.shape}')
        print(f'Image Shape : {h_img}x{w_img}x{c_img}')
        self.total_pixels = self.rgb_images.size(1)
        print(f"[INFO] image size: {self.img_res[1]}x{self.img_res[0]}, {self.total_pixels} pixels in total")
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy()
        self.uv = torch.from_numpy(uv).float()
        self.uv = self.uv.reshape(2, -1).transpose(1, 0) # (h*w, 2)

        mask_dir = '{0}/mask'.format(self.instance_dir)
        self.use_mask = use_mask
        if self.use_mask:
            if os.path.exists(mask_dir):
                mask_paths = sorted(utils.glob_imgs(mask_dir))
                # assert len(mask_paths) == self.n_images
                self.mask_images = []
                for path in mask_paths:
                    mask = rend_util.load_mask(path)
                    mask = mask.reshape(-1, 1)
                    self.mask_images.append(torch.from_numpy(mask).float())
                self.mask_images = torch.stack(self.mask_images, 0)
            else:
                print("[INFO] No existing mask image, use one mask as default")
                self.mask_images = torch.ones(self.n_images, self.total_pixels, 1, dtype=torch.float)
        
        lmask_dir = '{0}/light_mask'.format(self.instance_dir)
        self.use_lightmask = use_lightmask and os.path.exists(lmask_dir)
        if self.use_lightmask:
            lmask_paths = sorted(utils.glob_imgs(lmask_dir))
            self.lightmask_images = []
            for path in lmask_paths:
                lmask = rend_util.load_mask(path)
                lmask = lmask.reshape(-1, 1)
                self.lightmask_images.append(torch.from_numpy(lmask).float())
            self.lightmask_images = torch.stack(self.lightmask_images, 0)
        
        depth_dir = '{0}/depth'.format(self.instance_dir)
        print('depth dir : ', depth_dir)
        self.use_depth = use_depth and os.path.exists(depth_dir)
        print('Use depth : ', self.use_depth)
        self.use_bubble = use_bubble and os.path.exists(depth_dir)
        if self.use_depth or self.use_bubble:
            self.depth_images = []
            self.depth_masks = []
            self.pointcloud = [] # pointcloud for bubble loss, unprojected from depth and poses
            self.pointlinks = [] # link from pixel index to pointcloud index, value -1 when the pixel is invalid at pointcloud
            self.pixlinks = [] # link from pointcloud index to pixel index
            depth_paths = sorted(utils.glob_depths(depth_dir))
            print('Get depth_path ', depth_paths)
            n_points = 0
            if kwargs.get('noise_scale', 0.0) > 0:
                print(f"[INFO] Ablation study: using noise scale {kwargs.get('noise_scale')}")
            for scale_mat, depth_path, intrinsics, pose, i in tzip(scale_mats, depth_paths, self.intrinsics_all, self.pose_all, range(len(self.pose_all))):
                depth = rend_util.load_depth(depth_path)
                depth = torch.from_numpy(depth.reshape(-1)).float()
                depth = depth / scale_mat[2,2]
                valid_indices = torch.where((depth > 1e-3) & (depth < 6))[0]
                if i == 0 and scale_mat[2,2] != 1:
                    print(f"[INFO] Depth scaled by {scale_mat[2,2]:.2f}")
                depth_mask = torch.zeros([self.total_pixels], dtype=torch.bool)
                depth_mask[valid_indices] = True
                # if self.use_depth:
                if (noise_scale := kwargs.get('noise_scale', 0.0)) > 0:
                    depth = rend_util.add_depth_noise(depth, depth_mask.float(), noise_scale)
                self.depth_images.append(depth)
                self.depth_masks.append(depth_mask)
                if self.use_bubble:
                    pointlink = -torch.ones([self.total_pixels], dtype=torch.long)
                    pointlink[depth_mask] = torch.arange(0, len(valid_indices), dtype=torch.long) + n_points
                    pixlink = torch.arange(i * self.total_pixels, (i + 1) * self.total_pixels, dtype=torch.long)[depth_mask]
                    n_points += len(valid_indices)
                    self.pointlinks.append(pointlink)
                    self.pixlinks.append(pixlink)
                    self.pointcloud.append(rend_util.depth_to_world(self.uv, intrinsics, pose, depth, depth_mask))

            self.depth_images = torch.stack(self.depth_images, 0)
            self.depth_masks = torch.stack(self.depth_masks, 0)
            if self.use_bubble:
                self.pointcloud = torch.cat(self.pointcloud, 0)
                self.pointlinks = torch.cat(self.pointlinks, 0)
                self.pixlinks = torch.cat(self.pixlinks, 0)
                self.pointcloud = self.pointcloud[:,:3] / self.pointcloud[:,3:]
                self.pdf_prune = kwargs.get('pdf_prune', 0)
                self.pdf_max = kwargs.get('pdf_max', None)
                print(f"[INFO] PDF clamped to {self.pdf_prune}")
        
        normal_dir = '{0}/pred_normal'.format(self.instance_dir)
        dino_dir = '{0}/dino'.format(self.instance_dir)
        self.use_normal = use_normal and os.path.exists(normal_dir)
        self.use_dino = use_jaco and os.path.exists(dino_dir)
        self.use_jaco = self.use_normal and use_jaco and os.path.exists(dino_dir)
        print(f'Use Normal : {self.use_normal}, Use Jaco : {self.use_jaco}, Use Dino : {self.use_dino}')
        if self.use_normal or self.use_jaco:
            self.normal_images = []
            self.normal_masks = []
            normal_paths = [os.path.join(normal_dir, train_file[:-4]+'.npz') for train_file in train_list]
            print('Normal Length : ', len(normal_paths))
            for pose, normal_path in tzip(self.pose_all, normal_paths):

                normal = np.load(normal_path)['arr_0']
                normal = torch.from_numpy(normal.reshape(-1, 3)).float()

                valid_indices = torch.where(torch.linalg.vector_norm(normal, dim=1) > 1e-3)[0]
                R = pose[:3,:3]
                normal = (R @ normal.T).T # convert normal from view space to world space
                normal = F.normalize(normal, dim=1, eps=1e-6)
                self.normal_images.append(normal)
                normal_mask = torch.zeros([self.total_pixels], dtype=torch.bool)
                normal_mask[valid_indices] = True
                self.normal_masks.append(normal_mask)
            self.normal_images = torch.stack(self.normal_images, 0)
            self.normal_masks = torch.stack(self.normal_masks, 0)
            print('Normal Images : ', self.normal_images.shape)

        if self.use_dino:
            dino_paths = [os.path.join(dino_dir, train_file[:-4]+'.pth') for train_file in train_list]
            print('Dino length : ', len(dino_paths))
            
            dino_load_bar = tqdm(total=len(dino_paths))
            self.dino_torch = torch.zeros((len(dino_paths), h_img, w_img, 384), device='cpu')
            for idx, dino_name in enumerate(dino_paths):
                dino_single_torch = torch.load(dino_name).cpu().reshape(89, 119, 384)
                dino_single_np = dino_single_torch.cpu().numpy()
                # dino_single_resize = cv.resize(dino_single_np, (w_img, h_img), interpolation=cv.INTER_AREA) # for debug
                dino_single_resize = vit_feature_resize(dino_single_np, half_kernel_size=4, stride=4) # for accuracy
                self.dino_torch[idx] = torch.from_numpy(dino_single_resize).cpu()
                dino_load_bar.update(1)
                del dino_single_torch
                del dino_single_resize
                del dino_single_np
            self.dino_torch = self.dino_torch.reshape(*self.rgb_images.shape[:-1], -1)
            print('Dino Torch ', self.dino_torch.shape)

        self.dino_min = dino_min
        self.normal_min = normal_min
        print(f'Dino Min : {self.dino_min}; Normal Min : {self.normal_min}')

    def __len__(self):
        return self.n_images * self.total_pixels

    def __getitem__(self, idx):
        pidx = idx % self.total_pixels
        tidx = idx
        idx = idx // self.total_pixels
        sample = {
            "uv": self.uv[pidx].unsqueeze(0),
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        ground_truth = {
            "rgb": self.rgb_images[idx][pidx]
        }
        if self.use_mask:
            ground_truth['mask'] = self.mask_images[idx][pidx]
        if self.use_lightmask:
            ground_truth['light_mask'] = self.lightmask_images[idx][pidx]
        if self.use_depth or self.use_bubble:
            ground_truth['depth'] = self.depth_images[idx][pidx]
            ground_truth['depth_mask'] = self.depth_masks[idx][pidx]
        if self.use_normal:
            ground_truth['normal'] = self.normal_images[idx][pidx]
            ground_truth['normal_mask'] = self.normal_masks[idx][pidx]
        if self.use_jaco:
            ground_truth['dino_info'] = self.dino_torch[idx][pidx]
            

        return tidx, idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])

                # print(entry[0].keys())
                if 'dino_info' in entry[0].keys():
                    
                    dino_info = ret['dino_info']
                    # print('Dino Info : ', dino_info.shape)
                    dino_info_norm = dino_info / (torch.linalg.norm(dino_info, dim=-1).unsqueeze(-1) + 1e-7)
                    dino_info_cos = dino_info_norm @ torch.transpose(dino_info_norm, 0, 1)
                    dino_grid = dino_info_cos > self.dino_min # [bs, bs]

                    normal_info = ret['normal']
                    normal_info_norm = normal_info / (torch.linalg.norm(normal_info, dim=-1).unsqueeze(-1) + 1e-7)
                    normal_info_cos = normal_info_norm @ torch.transpose(normal_info_norm, 0, 1)
                    normal_grid = normal_info_cos > self.normal_min # [bs, bs]

                    jaco_info = dino_grid & normal_grid # [bs, bs]

                    ret['jaco_info'] = jaco_info
                    # print('Jacoinfo Shape : ', ret['jaco_info'].shape)
                    # print('RGB Shape : ', ret['rgb'].shape)

                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))
                # print('All Tupled + ', torch.LongTensor(entry).shape)

        return tuple(all_parsed)


class MaterialDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 scan_id=0,
                 downsample_train=1,
                 is_hdr=False,
                 **kwargs
                 ):
        self.sampling_idx = slice(None)

        self.instance_dir = os.path.join(data_dir, str(scan_id))

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        image_dir = '{0}/image'.format(self.instance_dir) if not is_hdr else '{0}/hdr'.format(self.instance_dir)
        self.is_hdr = is_hdr
        if is_hdr:
            print("[INFO] Using HDR image")
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras_normalize.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all, 0)
        self.pose_all = torch.stack(self.pose_all, 0)

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path, is_hdr=is_hdr)
            self.img_res = [rgb.shape[1], rgb.shape[2]]
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
        self.rgb_images = torch.stack(self.rgb_images, 0)
        self.total_pixels = self.rgb_images.size(1)

        mask_dir = '{0}/mask'.format(self.instance_dir)
        self.use_mask = os.path.exists(mask_dir)
        if self.use_mask:
            mask_paths = sorted(utils.glob_imgs(mask_dir))
            # assert len(mask_paths) == self.n_images
            self.mask_images = []
            for path in mask_paths:
                mask = rend_util.load_mask(path)
                mask = mask.reshape(-1, 1)
                self.mask_images.append(torch.from_numpy(mask).float())
            self.mask_images = torch.stack(self.mask_images, 0)
        
        lmask_dir = '{0}/light_mask'.format(self.instance_dir)
        self.use_lightmask = os.path.exists(lmask_dir)
        if self.use_lightmask:
            print("[INFO] Light mask detected")
            lmask_paths = sorted(utils.glob_imgs(lmask_dir))
            self.lightmask_images = []
            for path in lmask_paths:
                lmask = rend_util.load_mask(path)
                lmask = lmask.reshape(-1, 1)
                self.lightmask_images.append(torch.from_numpy(lmask).float())
            self.lightmask_images = torch.stack(self.lightmask_images, 0)
        
        if downsample_train > 1:
            old_res = (self.img_res[0], self.img_res[1])
            self.rgb_images = self.rgb_images.transpose(1, 2).reshape(-1, 3, *old_res)
            self.img_res[0] //= downsample_train
            self.img_res[1] //= downsample_train
            self.total_pixels = self.img_res[0] * self.img_res[1]
            self.rgb_images = F.interpolate(self.rgb_images, self.img_res, mode='area')
            self.rgb_images = self.rgb_images.reshape(-1, 3, self.total_pixels).transpose(1, 2)
            self.intrinsics_all = self.intrinsics_all.clone()
            self.intrinsics_all[:,0,0] /= downsample_train
            self.intrinsics_all[:,1,1] /= downsample_train
            self.intrinsics_all[:,0,2] /= downsample_train
            self.intrinsics_all[:,1,2] /= downsample_train
            if self.use_mask:
                self.mask_images = self.mask_images.transpose(1, 2).reshape(-1, 1, *old_res)
                self.mask_images = F.interpolate(self.mask_images, self.img_res, mode='area')
                self.mask_images = self.mask_images.reshape(-1, 1, self.total_pixels).transpose(1, 2)
            if self.use_lightmask:
                self.lightmask_images = self.lightmask_images.transpose(1, 2).reshape(-1, 1, *old_res)
                self.lightmask_images = F.interpolate(self.lightmask_images, self.img_res, mode='area')
                self.lightmask_images[self.lightmask_images > 0] = 1
                self.lightmask_images = self.lightmask_images.reshape(-1, 1, self.total_pixels).transpose(1, 2)

        print(f"[INFO] image size: {self.img_res[1]}x{self.img_res[0]}, {self.total_pixels} pixels in total")
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy()
        self.uv = torch.from_numpy(uv).float()
        self.uv = self.uv.reshape(2, -1).transpose(1, 0) # (h*w, 2)

    def __len__(self):
        return self.n_images * self.total_pixels

    def __getitem__(self, idx):
        pidx = idx % self.total_pixels
        tidx = idx
        idx = idx // self.total_pixels
        sample = {
            "uv": self.uv[pidx].unsqueeze(0),
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        ground_truth = {
            "rgb": self.rgb_images[idx][pidx]
        }
        if self.use_mask:
            ground_truth['mask'] = self.mask_images[idx][pidx]
        if self.use_lightmask:
            ground_truth['light_mask'] = self.lightmask_images[idx][pidx]
        return tidx, idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)