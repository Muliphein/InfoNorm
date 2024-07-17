import os
import torch
import numpy as np

import torch.nn.functional as F
from tqdm.contrib import tzip, tenumerate
import utils.general as utils
from utils import rend_util
from tqdm import tqdm
import pickle
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

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 dino_min=-1,
                 normal_min=-1,
                 **kwargs
                 ):

        self.instance_dir = os.path.join(data_dir, scan_id)
        self.data_dir = self.instance_dir

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"
        self.split_name = kwargs['split']
        with open(os.path.join(self.data_dir, self.split_name), 'rb') as file:
            serialized_list = file.read()
            train_list, eval_list = pickle.loads(serialized_list)

        image_paths = [os.path.join(f'{self.instance_dir}/image', train_file) for train_file in train_list]

        self.sampling_idx = None
        self.n_images = len(image_paths)
        print(f'Total Images : {len(image_paths)}')

        self.cam_file = '{0}/cameras_sphere.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)

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

        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            c_img, h_img, w_img = rgb.shape
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())
            # print('Single RGB Shape : ', torch.from_numpy(rgb).float().shape)

        
        self.dino_min = dino_min
        self.normal_min = normal_min
        print(f'Dino Min : {self.dino_min}; Normal Min : {self.normal_min}')
        self.use_jaco = False
        if self.dino_min > 0 and self.normal_min > 0:
            self.use_jaco = True
            print("Use Jaco")

        normal_dir = '{0}/pred_normal'.format(self.instance_dir)
        dino_dir = '{0}/dino'.format(self.instance_dir)
        if self.use_jaco:
            self.normal_images = []
            normal_paths = [os.path.join(normal_dir, train_file[:-4]+'.npz') for train_file in train_list]
            print('Normal Length : ', len(normal_paths))
            for pose, normal_path in tzip(self.pose_all, normal_paths):
                normal = np.load(normal_path)['arr_0']
                normal = torch.from_numpy(normal.reshape(-1, 3)).float()
                normal = F.normalize(normal, dim=1, eps=1e-6)
                self.normal_images.append(normal)

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
            self.dino_torch = self.dino_torch.reshape(len(dino_paths), h_img * w_img,-1)
            print('Dino Torch ', self.dino_torch.shape)




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
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:

            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["uv"] = uv[self.sampling_idx, :]
            if self.use_jaco:
                normal_info = self.normal_images[idx][self.sampling_idx, :]
                dino_info = self.dino_torch[idx][self.sampling_idx, :]

                # print('Dino Info : ', dino_info.shape)
                dino_info_norm = dino_info / (torch.linalg.norm(dino_info, dim=-1).unsqueeze(-1) + 1e-7)
                dino_info_cos = dino_info_norm @ torch.transpose(dino_info_norm, 0, 1)
                dino_grid = dino_info_cos > self.dino_min # [bs, bs]

                # print('Normal Info : ', normal_info.shape)
                normal_info_norm = normal_info / (torch.linalg.norm(normal_info, dim=-1).unsqueeze(-1) + 1e-7)
                normal_info_cos = normal_info_norm @ torch.transpose(normal_info_norm, 0, 1)
                normal_grid = normal_info_cos > self.normal_min # [bs, bs]

                jaco_info = dino_grid & normal_grid # [bs, bs]
                # print('Jaco Info : ', jaco_info.shape)

                ground_truth['jaco_info'] = jaco_info

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
