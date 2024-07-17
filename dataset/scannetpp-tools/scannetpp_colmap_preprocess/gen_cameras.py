import numpy as np
import trimesh
import cv2 as cv
import sys
import os
from glob import glob
import pickle
import shutil
from tqdm import tqdm

def export_traintest_split(split_file, image_name_list, div=2):
    if div == 1:
        print(f'Split in Div {div}')
        train_name_list = image_name_list[:]
        eval_name_list = image_name_list[:1]
        print(f"Train images #{len(train_name_list)}, Eval images #{len(eval_name_list)}")
        serialized_list = pickle.dumps([train_name_list, eval_name_list])
        with open(split_file, 'wb') as file:
            file.write(serialized_list)
    else:
        assert div>=2, "Need set the division > 2"
        print(f'Split in Div {div}')
        train_name_list = image_name_list[::div]
        eval_name_list = image_name_list[div//2::div]
        eval_name_list = eval_name_list[::2]
        print(f"Train images #{len(train_name_list)}, Eval images #{len(eval_name_list)}")
        serialized_list = pickle.dumps([train_name_list, eval_name_list])
        with open(split_file, 'wb') as file:
            file.write(serialized_list)


if __name__ == '__main__':
    work_dir = sys.argv[1]
    copy_dir = sys.argv[2]
    final_width = float(sys.argv[3])
    print('Final Width == ', final_width)
    if final_width != -1:
        final_height = float(sys.argv[4])
    
    poses_hwf = np.load(os.path.join(work_dir, 'poses.npy')) # n_images, 3, 5
    poses_raw = poses_hwf[:, :, :4]
    hwf = poses_hwf[:, :, 4]
    pose = np.diag([1.0, 1.0, 1.0, 1.0])
    pose[:3, :4] = poses_raw[0]

    cam_dict = dict()
    n_images = len(poses_raw)

    # Convert space
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0

    h0, w0, f0 = hwf[0, 0], hwf[0, 1], hwf[0, 2]
    aimh = h0
    aimw = w0
    if final_width != -1:
        scaled = min(h0 // final_height, w0 // final_width)
        aimh = scaled * final_height
        aimw = scaled * final_width
    print(f'Aim At {aimh}x{aimw}')
        

    for i in range(n_images):
        pose = np.diag([1.0, 1.0, 1.0, 1.0]).astype(np.float32)
        pose[:3, :4] = poses_raw[i]
        pose = pose @ convert_mat
        h, w, f = hwf[i, 0], hwf[i, 1], hwf[i, 2]
        assert h == h0 and w == w0 and f == f0
        scale = 1
        if (final_width != -1):
            scale = final_width / aimw
        intrinsic = np.diag([f*scale, f*scale, 1.0, 1.0]).astype(np.float32)
        intrinsic[0, 2] = (aimw*scale - 1) * 0.5
        intrinsic[1, 2] = (aimh*scale - 1) * 0.5
        
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)
        cam_dict['camera_mat_{:0>3d}'.format(i)] = intrinsic
        cam_dict['camera_mat_inv_{:0>3d}'.format(i)] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{:0>3d}'.format(i)] = world_mat
        cam_dict['world_mat_inv_{:0>3d}'.format(i)] = np.linalg.inv(world_mat)


    pcd = trimesh.load(os.path.join(work_dir, 'mesh.ply'))
    vertices = pcd.vertices
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    os.makedirs(copy_dir, exist_ok=True)
    
    np.savetxt(os.path.join(copy_dir, "trans_n2w.txt"), scale_mat, fmt='%.10f')
    
    for i in range(n_images):
        cam_dict['scale_mat_{:0>3d}'.format(i)] = scale_mat
        cam_dict['scale_mat_inv_{:0>3d}'.format(i)] = np.linalg.inv(scale_mat)

    out_dir = os.path.join(work_dir, 'preprocessed')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'mask'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'image_denoised'), exist_ok=True)

    image_list = glob(os.path.join(work_dir, 'images/*.JPG'))
    image_list.sort()
    
    
    
    name_list = []

    with tqdm(total= len(image_list), desc='image process') as tbar:
        for i, image_path in enumerate(image_list):
            name_list.append('{:0>3d}.png'.format(i))
            img = cv.imread(image_path)
            if final_width != -1:
                height, width = img.shape[:2]
                roi_x = (width - aimw) // 2
                roi_y = (height - aimh) // 2
                # print(f'Roi {roi_y}-{round(roi_y+aimh)} x {roi_x}-{round(roi_x+aimw)}')
                img = img[round(roi_y):round(roi_y+aimh), round(roi_x):round(roi_x+aimw)]
                img = cv.resize(img, (round(final_width), round(final_height)))
                
            arg=[10, 10, 5, 15]
            image = cv.fastNlMeansDenoisingColored(
                img.astype(np.uint8),
                None,
                h = arg[0], 
                hColor = arg[1], 
                templateWindowSize = arg[2], 
                searchWindowSize = arg[3],
            )
                
            cv.imwrite(os.path.join(out_dir, 'image_denoised', '{:0>3d}.png'.format(i)), image)
            cv.imwrite(os.path.join(out_dir, 'image', '{:0>3d}.png'.format(i)), img)
            cv.imwrite(os.path.join(out_dir, 'mask', '{:0>3d}.png'.format(i)), np.ones_like(img) * 255)
            tbar.update(1)

    split_file = os.path.join(out_dir, "split_sub1.pkl") # 300
    export_traintest_split(split_file, name_list, div=1)
    split_file = os.path.join(out_dir, "split_sub3.pkl") # 100
    export_traintest_split(split_file, name_list, div=3)
    split_file = os.path.join(out_dir, "split_sub5.pkl") # 60
    export_traintest_split(split_file, name_list, div=5)
    split_file = os.path.join(out_dir, "split_sub10.pkl") # 30
    export_traintest_split(split_file, name_list, div=10)
    split_file = os.path.join(out_dir, "split_sub20.pkl") # 30
    export_traintest_split(split_file, name_list, div=20)
    split_file = os.path.join(out_dir, "split_sub30.pkl") # 10
    export_traintest_split(split_file, name_list, div=30)
    
    np.savez(os.path.join(out_dir, 'cameras_sphere.npz'), **cam_dict)
    
    shutil.copy2(os.path.join(work_dir, "mesh.ply"), os.path.join(copy_dir, "mesh.ply"))
    shutil.copy2(os.path.join(work_dir, "preprocessed", "cameras_sphere.npz"), os.path.join(copy_dir, "cameras_sphere.npz"))
    shutil.copy2(os.path.join(work_dir, "preprocessed", "split_sub1.pkl"), os.path.join(copy_dir, "split_sub1.pkl"))
    shutil.copy2(os.path.join(work_dir, "preprocessed", "split_sub3.pkl"), os.path.join(copy_dir, "split_sub3.pkl"))
    shutil.copy2(os.path.join(work_dir, "preprocessed", "split_sub5.pkl"), os.path.join(copy_dir, "split_sub5.pkl"))
    shutil.copy2(os.path.join(work_dir, "preprocessed", "split_sub10.pkl"), os.path.join(copy_dir, "split_sub10.pkl"))
    shutil.copy2(os.path.join(work_dir, "preprocessed", "split_sub20.pkl"), os.path.join(copy_dir, "split_sub20.pkl"))
    shutil.copy2(os.path.join(work_dir, "preprocessed", "split_sub30.pkl"), os.path.join(copy_dir, "split_sub30.pkl"))
    
    try:
        shutil.rmtree(os.path.join(copy_dir, "image"))
    except:
        print('No Delete')
    shutil.copytree(os.path.join(work_dir, "preprocessed", "image"), os.path.join(copy_dir, "image"))
    
    try:
        shutil.rmtree(os.path.join(copy_dir, "image_denoised"))
    except:
        print('No Denoised Delete')
    shutil.copytree(os.path.join(work_dir, "preprocessed", "image_denoised"), os.path.join(copy_dir, "image_denoised"))
    
    shutil.rmtree(os.path.join(work_dir, "preprocessed"))
    # os.remove(os.path.join(work_dir, "poses.npy"))
    # os.remove(os.path.join(work_dir, "sparse_points.ply"))
    
    
    print('Process done!')
