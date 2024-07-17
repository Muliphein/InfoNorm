import argparse
import numpy as np
import os
from glob import glob
import trimesh
import cv2 as cv
from PIL import Image
import shutil
import pickle

#* replica camera intrinsic
REPLICA_FOCAL = 320.0
scale = 1.0

def export_images(max_length, image_list, output_image_list, resize_method = cv.INTER_AREA):
    image_h, image_w = cv.imread(image_list[0]).shape[:2]
    print(f'Image Shape : {image_h}x{image_w}')
    scale = max_length / max(image_h, image_w)
    image_new_h, image_new_w = round(image_h * scale), round(image_w * scale)
    print(f'Picture Scale : {scale}')
    for i in range(len(image_list)):
        #! Pillow Or Opencv
        #? Refer to https://github.com/python-pillow/Pillow/issues/2718
        output_image_path = output_image_list[i]

        #* pillow
        # image = Image.open(image_list[i])
        # image = image.resize((image_new_w, image_new_h))
        # image.save(output_image_path)

        #* opencv
        image = cv.imread(image_list[i])
        image = cv.resize(image, (image_new_w, image_new_h), resize_method)
        cv.imwrite(output_image_path, image)

def export_camera_npz(traj_file, camera_npz_file, intrinsic, gt_mesh_file, image_w, image_h, image_name_list):
    ts_full =  np.loadtxt(traj_file, delimiter=" ").reshape(-1, 4, 4)
    convert_mat = np.zeros([4, 4], dtype=np.float32)
    convert_mat[0, 1] = 1.0
    convert_mat[1, 0] = 1.0
    convert_mat[2, 2] =-1.0
    convert_mat[3, 3] = 1.0
    cam_dict = {}
    for i in range(ts_full.shape[0]):
        pose = ts_full[i]
        # must switch to [-u, r, -t] from [r, -u, t]
        pose = np.concatenate([pose[:, 1:2], pose[:, 0:1], -pose[:,2:3], pose[:,3:4]], 1)
        pose = pose @ convert_mat
        w2c = np.linalg.inv(pose)
        world_mat = intrinsic @ w2c
        world_mat = world_mat.astype(np.float32)

        cam_dict['camera_mat_{}'.format(image_name_list[i])] = intrinsic
        cam_dict['camera_mat_inv_{}'.format(image_name_list[i])] = np.linalg.inv(intrinsic)
        cam_dict['world_mat_{}'.format(image_name_list[i])] = world_mat
        cam_dict['world_mat_inv_{}'.format(image_name_list[i])] = np.linalg.inv(world_mat)

    pcd = trimesh.load(gt_mesh_file)
    vertices = pcd.vertices
    
    bbox_max = np.max(vertices, axis=0)
    bbox_min = np.min(vertices, axis=0)
    center = (bbox_max + bbox_min) * 0.5
    radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max() * 1.1
    scale_mat = np.diag([radius, radius, radius, 1.0]).astype(np.float32)
    scale_mat[:3, 3] = center

    for i in range(ts_full.shape[0]):
        cam_dict['scale_mat_{}'.format(image_name_list[i])] = scale_mat
        cam_dict['scale_mat_inv_{}'.format(image_name_list[i])] = np.linalg.inv(scale_mat)

    np.savez(camera_npz_file, **cam_dict)

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

def convert_data(args):
    #* Check the Legal

    file_list = os.listdir(args.input_path)
    assert "traj_w_c.txt" in file_list, "No TRAJ File"
    assert "rgb" in file_list, "No RGB File"

    #* origin name and convert name
    input_image_list = sorted(glob(os.path.join(args.input_path, "rgb", "*.png")), key=lambda x: int(os.path.basename(x)[4:-4]))
    image_name_list = [os.path.basename(x)[4:-4].zfill(3) for x in input_image_list]
    output_image_folder = os.path.join(args.output_path, "image")
    os.makedirs(output_image_folder, exist_ok=True)
    output_image_list = [os.path.join(output_image_folder, image_name+'.png') for image_name in image_name_list]
    if len(input_image_list) != len(os.listdir(output_image_folder)):
        export_images(args.max_length, input_image_list, output_image_list)
    else:
        print('Skip Image Copy')

    image_h, image_w = cv.imread(input_image_list[0]).shape[:2]
    scale = args.max_length / max(image_h, image_w)
    image_new_h, image_new_w = round(image_h * scale), round(image_w * scale)
    print(f'Image Shape : {image_h}x{image_w} -> {image_new_h}x{image_new_w}. Scale : {scale}')

    #* origin class image and convert name
    input_class_list = sorted(glob(os.path.join(args.input_path, "semantic_class", "semantic_class_*.png")), key=lambda x: int(os.path.basename(x)[15:-4]))
    print(f'Input Class list : {input_class_list[:5]} ... {input_class_list[-5:]}')
    class_name_list = [os.path.basename(x)[15:-4].zfill(3) for x in input_class_list]
    output_class_folder = os.path.join(args.output_path, "semantic")
    os.makedirs(output_class_folder, exist_ok=True)
    output_class_list = [os.path.join(output_class_folder, image_name+'.png') for image_name in class_name_list]
    if len(input_class_list) != len(os.listdir(output_class_folder)):
        export_images(args.max_length, input_class_list, output_class_list, cv.INTER_NEAREST)
    else:
        print('Skip Semantic Copy')



    camera_npz_file = os.path.join(args.output_path, "cameras_sphere.npz")
    traj_file = os.path.join(args.input_path, 'traj_w_c.txt')
    intrinsic = np.array([
        [REPLICA_FOCAL*scale, 0.0, (image_w-1)/2 * scale, 0.0],
        [0.0, REPLICA_FOCAL*scale, (image_h-1)/2 * scale, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    print('New Intrinsic : ', intrinsic)
    export_camera_npz(traj_file, camera_npz_file, intrinsic, args.gtply_path, image_new_w, image_new_h, image_name_list)

    shutil.copy2(args.gtply_path, os.path.join(args.output_path, "mesh.ply"))

    split_file = os.path.join(args.output_path, "split_sub1.pkl")
    export_traintest_split(split_file, [image_name+'.png' for image_name in image_name_list], div=1)
    split_file = os.path.join(args.output_path, "split_sub10.pkl")
    export_traintest_split(split_file, [image_name+'.png' for image_name in image_name_list], div=10)
    split_file = os.path.join(args.output_path, "split_sub20.pkl")
    export_traintest_split(split_file, [image_name+'.png' for image_name in image_name_list], div=20)
    split_file = os.path.join(args.output_path, "split_sub40.pkl")
    export_traintest_split(split_file, [image_name+'.png' for image_name in image_name_list], div=40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input folder", required=True, type=str)
    parser.add_argument("--output_path", help="Output folder", required=True, type=str)
    parser.add_argument("--gtply_path", help="Ground truth mesh", required=True, type=str)
    parser.add_argument("--max_length", help="Max length of the output picture", default=400, type=int)
    args = parser.parse_args()
    print(f'Convert Data From {args.input_path} to {args.output_path}')
    convert_data(args)
    
    pass