import argparse
import numpy as np
import os
from glob import glob
import trimesh
import cv2 as cv
from PIL import Image
import shutil
import pickle
from tqdm import tqdm
import random

#* replica camera intrinsic
REPLICA_FOCAL = 320.0
scale = 1.0

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

def export_images(max_length, image_list, output_image_list):
    if str(image_list[0]).endswith('.png') or str(image_list[0]).endswith('.jpg'):
        image_h, image_w, image_c = cv.imread(image_list[0]).shape[:3]
    elif str(image_list[0]).endswith('.npz'):
        image_h, image_w, image_c = np.load(image_list[0])['arr_0'].shape[:3]
    # print(f'Image Shape : {image_h}x{image_w}x{image_c}', end='')
    scale = max_length / max(image_h, image_w)
    image_new_h, image_new_w = round(image_h * scale), round(image_w * scale)
    # print(f' -> {image_new_h}x{image_new_w}x{image_c}', end='')

    # print(f'Picture Scale : {scale}')
    for i in range(len(image_list)):

        #! Different between opencv and pillow
        #? Refer to https://github.com/python-pillow/Pillow/issues/2718
        output_image_path = output_image_list[i]

        #* pillow
        # if str(image_list[i]).endswith('.png') or str(image_list[i]).endswith('.jpg'):
        #     image = Image.open(image_list[i])
        # elif str(image_list[i]).endswith('.npz'):
        #     image = np.load(image_list[i])['arr_0']
        #     image = Image.fromarray(image)
        # print(f'Image Shape : {np.array(image).shape}', end='')
        # image = image.resize((image_new_w, image_new_h))
        # print(f'-> {np.array(image).shape}')
        # if str(image_list[i]).endswith('.png') or str(image_list[i]).endswith('.jpg'):
        #     image.save(output_image_path)
        # elif str(image_list[i]).endswith('.npz'):
        #     image = np.array(image)
        #     np.savez(output_image_path, image)

        #* opencv
        if str(image_list[i]).endswith('.png') or str(image_list[i]).endswith('.jpg'):
            image = cv.imread(image_list[i])
            # print(f'Image Shape : {np.array(image).shape}', end='')
            image = cv.resize(image, (image_new_w, image_new_h))
            # print(f'-> {np.array(image).shape}')
            cv.imwrite(output_image_path, image)

        elif str(image_list[i]).endswith('.npz'):
            image = np.load(image_list[i])['arr_0']
            # print(f'NPZ Shape : {np.array(image).shape}', end='')
            image = cv.resize(image, (image_new_w, image_new_h))
            # print(f'-> {np.array(image).shape}')
            np.savez(output_image_path, image)

def convert_camera_npz(input_camera_npz_file, output_camera_npz_file, scale, image_name_list):

    input_cam_dict = np.load(input_camera_npz_file)
    output_cam_dict = {}
    n_images = len(image_name_list)
    world_mat_counter = 0
    for file in input_cam_dict.files:
        if str(file).startswith('world_mat_') and not str(file).startswith('world_mat_inv_'):
            world_mat_counter += 1
    assert world_mat_counter == n_images, "Not Match in npz and images"

    for i in range(n_images):
        w2p = input_cam_dict[f'world_mat_{i}'][:3, :4]
        scale_mat = input_cam_dict[f'scale_mat_{i}']
        intr, c2w = load_K_Rt_from_P(None, w2p)
        # print(f'Intr Old : {intr} -> ', end = '')
        intr[0][0] = intr[0][0] * scale
        intr[1][1] = intr[1][1] * scale
        intr[0][2] = intr[0][2] * scale
        intr[1][2] = intr[1][2] * scale
        # print(f'Intr New : {intr} -> ')
        world_mat = intr @ np.linalg.inv(c2w)
        world_mat = world_mat.astype(np.float32)

        output_cam_dict['camera_mat_{}'.format(image_name_list[i])] = intr
        output_cam_dict['camera_mat_inv_{}'.format(image_name_list[i])] = np.linalg.inv(intr)
        output_cam_dict['world_mat_{}'.format(image_name_list[i])] = world_mat
        output_cam_dict['world_mat_inv_{}'.format(image_name_list[i])] = np.linalg.inv(world_mat)
        output_cam_dict['scale_mat_{}'.format(image_name_list[i])] = scale_mat
        output_cam_dict['scale_mat_inv_{}'.format(image_name_list[i])] = np.linalg.inv(scale_mat)


    np.savez(output_camera_npz_file, **output_cam_dict)

def export_traintest_split_aim(split_file, image_name_list, aim=50, max_eval=50):
    div = len(image_name_list) // aim - 1
    div = max(1, div)
    print(f'Split in Aim = {aim}')
    if div == 1:
        eval_name_list = random.sample(image_name_list, (len(image_name_list) + 4) // 5)
        train_name_list = list(set(image_name_list) - set(eval_name_list))

        eval_name_list = sorted(eval_name_list)
        train_name_list = sorted(train_name_list)
    else:
        train_name_list = image_name_list[::div]
        eval_name_list = image_name_list[div//2::div]
        eval_name_list = eval_name_list[::2]
        if len(eval_name_list) > max_eval:
            eval_name_list = random.sample(eval_name_list, max_eval)
        eval_name_list = sorted(eval_name_list)
        train_name_list = sorted(train_name_list)
    print(f"Train images #{len(train_name_list)}, Eval images #{len(eval_name_list)}")
    serialized_list = pickle.dumps([train_name_list, eval_name_list])
    with open(split_file, 'wb') as file:
        file.write(serialized_list)

def export_denoise_images(input_image_list, output_denoise_list, arg=[10, 10, 7, 21]):
    assert len(input_image_list) == len(output_denoise_list), "Denoise should have same length"
    tbar = tqdm(total=len(input_image_list), desc='Denoise Image')
    for i in range(len(input_image_list)):
        image = cv.imread(input_image_list[i])
        image = cv.fastNlMeansDenoisingColored(
            image.astype(np.uint8),
            None,
            h = arg[0], 
            hColor = arg[1], 
            templateWindowSize = arg[2], 
            searchWindowSize = arg[3],
        )
        cv.imwrite(output_denoise_list[i], image)
        tbar.update(1)

def convert_data(args):
    #* Check the Legal

    file_list = os.listdir(args.input_path)
    assert "cameras_sphere.npz" in file_list, "No Camera File"
    assert "image" in file_list, "No Image File"
    assert "pred_normal" in file_list, "No Normal File"

    #* origin name and convert name
    input_image_list = sorted(glob(os.path.join(args.input_path, "image", "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))
    image_name_list = [os.path.basename(x)[:-4].zfill(4) for x in input_image_list]

    # convert picture to small
    output_image_folder = os.path.join(args.output_path, "image")
    os.makedirs(output_image_folder, exist_ok=True)
    output_image_list = [os.path.join(output_image_folder, image_name+'.png') for image_name in image_name_list]

    if len(input_image_list) != len(os.listdir(output_image_folder)):
        export_images(args.max_length, input_image_list, output_image_list)
    else:
        print('Skip Image Copy')

    # convert Normal -> Scaled Normal
    output_normal_folder = os.path.join(args.output_path, "pred_normal")
    os.makedirs(output_normal_folder, exist_ok=True)
    input_normal_list = [os.path.join(args.input_path, "pred_normal", image_name+'.npz') for image_name in image_name_list]
    output_normal_list = [os.path.join(args.output_path, "pred_normal", image_name) for image_name in image_name_list]

    if len(input_image_list) != len(os.listdir(output_normal_folder)):
        export_images(args.max_length, input_normal_list, output_normal_list)
    else:
        print('Skip Normal Copy')

    # Make Image -> Scaled Image -> Denoised Scaled Image
    output_denoise_folder = os.path.join(args.output_path, "image_denoised")
    os.makedirs(output_denoise_folder, exist_ok=True)
    output_denoise_list = [os.path.join(output_denoise_folder, image_name+'.png') for image_name in image_name_list]
    
    if len(output_denoise_list) != len(os.listdir(output_denoise_folder)):
        export_denoise_images(output_image_list, output_denoise_list, arg=[10, 10, 5, 15])
    else:
        print('Skip Denoise Copy')

    # Image -> Denoised Image -> Scaled Denoised Image
    output_original_denoise_folder = os.path.join(args.output_path, "image_denoised_origin")
    os.makedirs(output_original_denoise_folder, exist_ok=True)
    output_denoise_ori_list = [os.path.join(output_original_denoise_folder, image_name+'.png') for image_name in image_name_list]
    if len(output_denoise_ori_list) != len(os.listdir(output_original_denoise_folder)):
        export_denoise_images(input_image_list, output_denoise_ori_list, arg=[10, 10, 7, 21])
    else:
        print('Skip Denoise OriginScale Copy')

    # convert picture to small
    output_denoised_ori_scale_folder = os.path.join(args.output_path, "image_denoised_ori_scale")
    os.makedirs(output_denoised_ori_scale_folder, exist_ok=True)
    output_dos_list = [os.path.join(output_denoised_ori_scale_folder, image_name+'.png') for image_name in image_name_list]

    if len(output_dos_list) != len(os.listdir(output_denoised_ori_scale_folder)):
        export_images(args.max_length, output_denoise_ori_list, output_dos_list)
    else:
        print('Skip Original Scale Denoised Image Copy')


    image_h, image_w = cv.imread(input_image_list[0]).shape[:2]
    scale = args.max_length / max(image_h, image_w)
    image_new_h, image_new_w = round(image_h * scale), round(image_w * scale)
    print(f'Image Shape : {image_h}x{image_w} -> {image_new_h}x{image_new_w}. Scale : {scale}')

    output_camera_npz_file = os.path.join(args.output_path, "cameras_sphere.npz")
    input_camera_npz_file = os.path.join(args.input_path, "cameras_sphere.npz")
    convert_camera_npz(
        input_camera_npz_file, output_camera_npz_file, scale, image_name_list
    )

    try:
        shutil.copy2(os.path.join(args.input_path, "trans_n2w.txt"), os.path.join(args.output_path, "trans_n2w.txt"))
    except:
        print("No Transtxt")

    try:
        shutil.copy2(os.path.join(args.input_path, "mesh.ply"), os.path.join(args.output_path, "mesh.ply"))
    except:
        print("No Gt Mesh")

    # split_file = os.path.join(args.output_path, "split_all.pkl")
    # export_traintest_split_aim(split_file, [image_name+'.png' for image_name in image_name_list], aim=len(image_name_list))
    # split_file = os.path.join(args.output_path, "split_30.pkl")
    # export_traintest_split_aim(split_file, [image_name+'.png' for image_name in image_name_list], aim=30)
    # split_file = os.path.join(args.output_path, "split_50.pkl")
    # export_traintest_split_aim(split_file, [image_name+'.png' for image_name in image_name_list], aim=50)
    split_file = os.path.join(args.output_path, "split_100.pkl")
    export_traintest_split_aim(split_file, [image_name+'.png' for image_name in image_name_list], aim=100)

if __name__ == "__main__":
    random.seed(3407)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="Input folder", required=True, type=str)
    parser.add_argument("--output_path", help="Output folder", required=True, type=str)
    parser.add_argument("--max_length", help="Max length of the output picture", default=400, type=int)
    args = parser.parse_args()
    print(f'Convert Data From {args.input_path} to {args.output_path}')
    convert_data(args)
    
    pass