import os
import torch
import torch.nn.functional as F
import numpy as np

import utils.general as utils
from utils import rend_util
from glob import glob
import cv2
import random
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

# class SceneDataset(torch.utils.data.Dataset):

#     def __init__(self,
#                  data_dir,
#                  img_res,
#                  scan_id=0,
#                  num_views=-1,  
#                  ):

#         self.instance_dir = os.path.join('../data', data_dir, 'scan{0}'.format(scan_id))

#         self.total_pixels = img_res[0] * img_res[1]
#         self.img_res = img_res

#         assert os.path.exists(self.instance_dir), "Data directory is empty"
        
#         self.num_views = num_views
#         assert num_views in [-1, 3, 6, 9]
        
#         self.sampling_idx = None

#         image_dir = '{0}/image'.format(self.instance_dir)
#         image_paths = sorted(utils.glob_imgs(image_dir))
#         self.n_images = len(image_paths)

#         self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
#         camera_dict = np.load(self.cam_file)
#         scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
#         world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

#         self.intrinsics_all = []
#         self.pose_all = []
#         for scale_mat, world_mat in zip(scale_mats, world_mats):
#             P = world_mat @ scale_mat
#             P = P[:3, :4]
#             intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
#             self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
#             self.pose_all.append(torch.from_numpy(pose).float())

#         self.rgb_images = []
#         for path in image_paths:
#             rgb = rend_util.load_rgb(path)
#             rgb = rgb.reshape(3, -1).transpose(1, 0)
#             self.rgb_images.append(torch.from_numpy(rgb).float())
            
#         # used a fake depth image and normal image
#         self.depth_images = []
#         self.normal_images = []

#         for path in image_paths:
#             depth = np.ones_like(rgb[:, :1])
#             self.depth_images.append(torch.from_numpy(depth).float())
#             normal = np.ones_like(rgb)
#             self.normal_images.append(torch.from_numpy(normal).float())
            
#     def __len__(self):
#         return self.n_images

#     def __getitem__(self, idx):
#         if self.num_views >= 0:
#             image_ids = [25, 22, 28, 40, 44, 48, 0, 8, 13][:self.num_views]
#             idx = image_ids[random.randint(0, self.num_views - 1)]
            
#         uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
#         uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
#         uv = uv.reshape(2, -1).transpose(1, 0)

#         sample = {
#             "uv": uv,
#             "intrinsics": self.intrinsics_all[idx],
#             "pose": self.pose_all[idx]
#         }

#         ground_truth = {
#             "rgb": self.rgb_images[idx],
#             "depth": self.depth_images[idx],
#             "normal": self.normal_images[idx],
#         }
        
#         if self.sampling_idx is not None:
#             ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
#             ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
#             ground_truth["mask"] = torch.ones_like(self.depth_images[idx][self.sampling_idx, :])
#             ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            
#             sample["uv"] = uv[self.sampling_idx, :]

#         return idx, sample, ground_truth

#     def collate_fn(self, batch_list):
#         # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
#         batch_list = zip(*batch_list)

#         all_parsed = []
#         for entry in batch_list:
#             if type(entry[0]) is dict:
#                 # make them all into a new dict
#                 ret = {}
#                 for k in entry[0].keys():
#                     ret[k] = torch.stack([obj[k] for obj in entry])
#                 all_parsed.append(ret)
#             else:
#                 all_parsed.append(torch.LongTensor(entry))

#         return tuple(all_parsed)

#     def change_sampling_idx(self, sampling_size):
#         if sampling_size == -1:
#             self.sampling_idx = None
#         else:
#             self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

#     def get_scale_mat(self):
#         return np.load(self.cam_file)['scale_mat_0']

import pickle

# Dataset with monocular depth and normal
class SceneDatasetDN(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 center_crop_type='xxxx',
                 use_mask=False,
                 num_views=-1,
                 split=None,
                 dino_min=None,
                 normal_min=None,
                 semantic_type='dino',
                 ):

        self.instance_dir = os.path.join('../../../dataset/processored', data_dir, scan_id)
        self.split_name = split
        self.data_dir = self.instance_dir
        self.semantic_type = semantic_type
        print('Semantic Type : ', self.semantic_type)

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        self.num_views = num_views
        assert num_views in [-1] # Define by split, not num_views
        
        print('Load Data from ', self.instance_dir)
        assert os.path.exists(self.instance_dir), "Data directory is empty"
        print(f"[INFO] Loading data from {self.instance_dir}")

        self.sampling_idx = None
        
        def glob_data(data_dir):
            data_paths = []
            data_paths.extend(glob(data_dir))
            data_paths = sorted(data_paths)
            return data_paths
        
        with open(os.path.join(self.data_dir, self.split_name), 'rb') as file:
            serialized_list = file.read()
            train_list, eval_list = pickle.loads(serialized_list)

        image_paths = [os.path.join(self.data_dir, 'omni_set', train_file[:-4]+'_rgb.png') for train_file in train_list]
        depth_paths = [os.path.join(self.data_dir, 'omni_set', train_file[:-4]+'_depth.npy') for train_file in train_list]
        normal_paths = [os.path.join(self.data_dir, 'omni_set', train_file[:-4]+'_normal.npy') for train_file in train_list]

        # mask is only used in the replica dataset as some monocular depth predictions have very large error and we ignore it
        if use_mask:
            assert False, "No Mask Plz"
            mask_paths = glob_data(os.path.join('{0}'.format(self.instance_dir), "*_mask.npy"))
        else:
            mask_paths = None

        self.n_images = len(image_paths)
        
        self.cam_file = '{0}/cameras_sphere.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%s' % pic_name[:-4]].astype(np.float32) for pic_name in train_list]
        world_mats = [camera_dict['world_mat_%s' % pic_name[:-4]].astype(np.float32) for pic_name in train_list]
        self.scale_mat = scale_mats[0]
        print('Scale Mat Shape :', self.scale_mat.shape)
        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            # print('Intri Before : ', intrinsics)
            # only crop 480*360 -> 360 * 360
            intrinsics[0][2] = (img_res[0]-1)*0.5
            intrinsics[1][2] = (img_res[1]-1)*0.5
            # print('Intri After : ', intrinsics)

            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            # print('RGB Shape : ', rgb.shape)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            
        self.depth_images = []
        self.normal_images = []

        for dpath, npath in zip(depth_paths, normal_paths):
            depth = np.load(dpath)
            self.depth_images.append(torch.from_numpy(depth.reshape(-1, 1)).float())
        
            normal = np.load(npath)
            normal = normal.reshape(3, -1).transpose(1, 0)
            # important as the output of omnidata is normalized
            normal = normal * 2. - 1.
            self.normal_images.append(torch.from_numpy(normal).float())

        # load mask
        self.mask_images = []
        if mask_paths is None:
            for depth in self.depth_images:
                mask = torch.ones_like(depth)
                self.mask_images.append(mask)
        else:
            for path in mask_paths:
                mask = np.load(path)
                self.mask_images.append(torch.from_numpy(mask.reshape(-1, 1)).float())

        self.use_dino = dino_min != None and normal_min != None
        if self.use_dino:
            if self.semantic_type == 'dino':
                dino_paths = [os.path.join(self.data_dir, 'dino', train_file[:-4]+'.pth') for train_file in train_list]
                print('Dino length : ', len(dino_paths))
                
                dino_load_bar = tqdm(total=len(dino_paths))
                self.dino_torch = torch.zeros((len(dino_paths), 360, 360, 384))
                for idx, dino_name in enumerate(dino_paths):
                    dino_single_torch = torch.load(dino_name).cpu().reshape(89, 119, 384) # [89, 119, 384]
                    dino_single_np = dino_single_torch.cpu().numpy()
                    dino_single_resize = cv.resize(dino_single_np, (480, 360), interpolation=cv.INTER_AREA) # [360, 480, 384]
                    # dino_single_resize = vit_feature_resize(dino_single_np, half_kernel_size=4, stride=4) # for accuracy
                    self.dino_torch[idx] = torch.from_numpy(dino_single_resize[:, 60:-60, :])
                    dino_load_bar.update(1)
                    del dino_single_torch
                    del dino_single_resize
                    del dino_single_np
                self.dino_torch = self.dino_torch.reshape(len(dino_paths), -1, 384)
                print('Dino Torch ', self.dino_torch.shape)
            elif self.semantic_type == 'sam':
                dino_paths = [os.path.join(self.data_dir, 'sam', i) for i in train_list]
                print('Dino length : ', len(dino_paths))
                dino_load_bar = tqdm(total=len(dino_paths))
                self.dino_torch = torch.zeros((len(dino_paths), 360, 360, 3))
                for idx, dino_name in enumerate(dino_paths):
                    dino_single_torch = cv.imread(dino_name) / 256.0
                    self.dino_torch[idx] = torch.from_numpy(dino_single_torch[:, 60:-60, :]).cpu()
                    dino_load_bar.update(1)
                    del dino_single_torch
                self.dino_torch = self.dino_torch.reshape(len(dino_paths), -1, 3)
                print('Dino Torch ', self.dino_torch.shape)
            else:
                raise NotImplementedError
        
        
        self.dino_min = dino_min
        self.normal_min = normal_min

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "pose": self.pose_all[idx]
        }
        
        ground_truth = {
            "rgb": self.rgb_images[idx],
            "depth": self.depth_images[idx],
            "mask": self.mask_images[idx],
            "normal": self.normal_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["full_rgb"] = self.rgb_images[idx]
            ground_truth["normal"] = self.normal_images[idx][self.sampling_idx, :]
            ground_truth["depth"] = self.depth_images[idx][self.sampling_idx, :]
            ground_truth["full_depth"] = self.depth_images[idx]
            ground_truth["mask"] = self.mask_images[idx][self.sampling_idx, :]
            ground_truth["full_mask"] = self.mask_images[idx]
         
            sample["uv"] = uv[self.sampling_idx, :]
            if self.use_dino:
                ground_truth['dino_info'] = self.dino_torch[idx][self.sampling_idx, :]
        return idx, sample, ground_truth

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
                
                if 'dino_info' in entry[0].keys():
                    
                    dino_info = ret['dino_info'].squeeze()
                    dino_mask = ~torch.all(dino_info == 0, dim=1)
                    # print('Dino Info : ', dino_info.shape)
                    dino_info_norm = dino_info / (torch.linalg.norm(dino_info, dim=-1).unsqueeze(-1) + 1e-7)
                    dino_info_cos = dino_info_norm @ torch.transpose(dino_info_norm, 0, 1)
                    dino_grid = dino_info_cos > self.dino_min # [bs, bs]

                    normal_info = ret['normal'].squeeze()
                    # print('Normal Info : ', normal_info.shape)
                    normal_info_norm = normal_info / (torch.linalg.norm(normal_info, dim=-1).unsqueeze(-1) + 1e-7)
                    normal_info_cos = normal_info_norm @ torch.transpose(normal_info_norm, 0, 1)
                    normal_grid = normal_info_cos > self.normal_min # [bs, bs]

                    jaco_info = dino_grid & normal_grid # [bs, bs]

                    ret['jaco_info'] = jaco_info
                    ret['dino_mask'] = dino_mask
                    # print('Jacoinfo Shape : ', ret['jaco_info'].shape)
                    # print('RGB Shape : ', ret['rgb'].shape)

                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
