# This file is borrowed from NeuS2 : https://github.com/19reborn/NeuS2
import json
from os.path import join
import numpy as np
import os
import cv2
import torch
from glob import glob
import shutil

import pickle

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
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
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.conf = conf

        self.data_dir = conf['data_dir']
        self.render_cameras_name = conf['render_cameras_name']
        self.object_cameras_name = conf['object_cameras_name']

        self.camera_outside_sphere = True
        self.scale_mat_scale = 1.1

        camera_dict = np.load(os.path.join(self.data_dir, self.render_cameras_name))
        self.camera_dict = camera_dict
        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'image/*.png')))
        self.n_images = len(self.images_lis)

        self.fname_lis = [os.path.basename(fname)[:-4] for fname in self.images_lis]
        # print(self.fname_lis)

        self.world_mats_np = [camera_dict['world_mat_%s' % (fname)].astype(np.float32) for fname in self.fname_lis]

        self.scale_mats_np = []
        self.scale_mats_np = [camera_dict['scale_mat_%s' % (fname)].astype(np.float32) for fname in self.fname_lis]

        self.intrinsics_all = []
        self.pose_all = []

        for scale_mat, world_mat in zip(self.scale_mats_np, self.world_mats_np):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intr, c2w = load_K_Rt_from_P(None, P)
            # c2w = c2w[:, 1:3] *= -1
            self.intrinsics_all.append(torch.from_numpy(intr).float())
            self.pose_all.append(torch.from_numpy(c2w).float())

        self.intrinsics_all = torch.stack(self.intrinsics_all)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]

        print('Load data: End')


def generate(dataset_name, base_par_dir, copy_image=False, is_downsample=False, downsample_scale=1, fixed_camera=True, wrong_camera=[]):
    assert is_downsample == False, "Not implemented"

    base_dir = os.path.join(base_par_dir, dataset_name)

    conf = {
        "data_dir": base_dir,
        "render_cameras_name": "cameras_sphere.npz",
        "object_cameras_name": "cameras_sphere.npz",
    }

    dataset = Dataset(conf)
    image_name = 'image'
    base_rgb_dir = join(base_dir,image_name)

    all_images = sorted(os.listdir(base_rgb_dir))
    print("#images:", len(all_images))

    image_sample = os.path.join(base_rgb_dir, all_images[0])
    image_sample = cv2.imread(image_sample)
    H, W = image_sample.shape[:2]
    print(f'H : {H}; W : {W}')

    # H, W = 360, 480

    base_rgb_dir = "image"
    # print("base_rgb_dir:", base_rgb_dir)

    output = {
        "w": W,
        "h": H,
        "aabb_scale": 1.0,
        "scale": 0.5,
        "offset": [ # neus: [-1,1] ngp[0,1]
            0.5,
            0.5,
            0.5
        ],
        "from_na": True,
    }

    output['frames'] = []
    camera_num = dataset.intrinsics_all.shape[0]
    for i in range(camera_num):
        rgb_dir = join('image', dataset.fname_lis[i]+'.png')
        ixt = dataset.intrinsics_all[i]
        one_frame = {}
        one_frame["file_path"] = rgb_dir
        one_frame["transform_matrix"] = dataset.pose_all[i].tolist()
        one_frame["intrinsic_matrix"] = ixt.tolist()
        output['frames'].append(one_frame)
    file_dir = join(base_dir, f'transforms.json')
    with open(file_dir,'w') as f:
        json.dump(output, f, indent=4)
    
    pkl_list = os.listdir(base_dir)
    
    for pkl_fname in pkl_list:
        if not str(pkl_fname).endswith('.pkl'):
            continue
        
        with open(os.path.join(base_dir, pkl_fname), 'rb') as file:
            serialized_list = file.read()
            train_list, eval_list = pickle.loads(serialized_list)
        new_output = {
            "w": W,
            "h": H,
            "aabb_scale": 1.0,
            "scale": 0.5,
            "offset": [ # neus: [-1,1] ngp[0,1]
                0.5,
                0.5,
                0.5
            ],
            "from_na": True,
        }
        new_output['frames'] = []
        for i in range(camera_num):
            rgb_dir = join('image', dataset.fname_lis[i]+'.png')
            ixt = dataset.intrinsics_all[i]
            one_frame = {}
            one_frame["file_path"] = rgb_dir
            one_frame["transform_matrix"] = dataset.pose_all[i].tolist()
            one_frame["intrinsic_matrix"] = ixt.tolist()
            if dataset.fname_lis[i]+'.png' in train_list:
                new_output['frames'].append(one_frame)
                
        file_dir = join(base_dir, f'{pkl_fname[:-4]}.json')
        with open(file_dir,'w') as f:
            json.dump(new_output, f, indent=4)


    


if __name__ == "__main__":
    base_par_dir = "processored/Replica/"
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    for dataset_name in os.listdir(base_par_dir):
        print("dataset_name:", dataset_name)
        generate(dataset_name, base_par_dir)
        # break
    
    