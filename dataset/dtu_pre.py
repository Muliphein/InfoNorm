import argparse
import numpy as np
import os
from glob import glob
import trimesh
import cv2 as cv
from PIL import Image
import pickle

#* replica camera intrinsic
REPLICA_FOCAL = 320.0
scale = 1.0

def export_images(max_length, image_list, output_image_list, ouptut_denoise_list, mask_image_list=None):
    image_h, image_w = cv.imread(image_list[0]).shape[:2]
    print(f'Image Shape : {image_h}x{image_w}')
    scale = max_length / max(image_h, image_w)
    image_new_h, image_new_w = round(image_h * scale), round(image_w * scale)
    print(f'Picture Scale : {scale}')
    for i in range(len(image_list)):
        #! Use Pillow for resize 
        #? Refer to https://github.com/python-pillow/Pillow/issues/2718
        output_image_path = output_image_list[i]
        
        #* pillow
        # image = Image.open(image_list[i])
        # image = image.resize((image_new_w, image_new_h))
        # image.save(output_image_path)

        
        
        #* opencv
        if mask_image_list is None:
            image = cv.imread(image_list[i])
            image = cv.resize(image, (image_new_w, image_new_h))
            cv.imwrite(output_image_path, image)
        else:
            
            image = cv.imread(image_list[i])
            mask = cv.imread(mask_image_list[i])
            print(f'image shape : {image.shape, image.max(), image.min()}, mask shape : {mask.shape, mask.max(), mask.min()}')
            image = cv.resize(image*(mask//255), (image_new_w, image_new_h))
            cv.imwrite(ouptut_denoise_list[i], cv.fastNlMeansDenoisingColored(image, None, h = 10, 
                                                                                            hColor = 10, 
                                                                                            templateWindowSize = 5, 
                                                                                            searchWindowSize = 15))
            cv.imwrite(output_image_path, image)

def convert_camera_npz(input_camera_npz_file, output_camera_npz_file, scale):
    in_cam_dict = np.load(input_camera_npz_file)

    out_cam_dict = {}
    for file in in_cam_dict.files:
        if str(file).startswith("world_mat_") and not str(file).startswith("world_mat_inv_"):
            name = str(file).split('_')[-1]
            # print(name)
            world_mat = in_cam_dict['world_mat_'+name]
            ins, c2w = load_K_Rt_from_P(None, world_mat[:3, :4])
            ins[0, 0] *= scale
            ins[1, 1] *= scale
            ins[0, 2] *= scale
            ins[1, 2] *= scale
            w2c = np.linalg.inv(c2w)
            world_mat_new = ins @ w2c
            out_cam_dict['world_mat_'+name.zfill(6)] = world_mat_new
            out_cam_dict['scale_mat_'+name.zfill(6)] = in_cam_dict['scale_mat_'+name]

    np.savez(output_camera_npz_file, **out_cam_dict)

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

def convert_data(input_path, output_path, max_length):
    print(glob(os.path.join(input_path, "image", "*.png")))
    #* origin name and convert name
    input_image_list = sorted(glob(os.path.join(input_path, "image", "*.png")), key=lambda x: int(os.path.basename(x)[:-4]))
    image_name_list = [os.path.basename(x)[:-4].zfill(3) for x in input_image_list]
    input_mask_name_list = [os.path.join(input_path, 'mask',image_name[3:]+'.png') for image_name in image_name_list]
    ouptut_denoise_list = [os.path.join(output_path, 'image_denoised',image_name+'.png') for image_name in image_name_list]
    os.makedirs(os.path.join(output_path, 'image_denoised'), exist_ok=True)
    output_image_folder = os.path.join(output_path, "image")
    os.makedirs(output_image_folder, exist_ok=True)
    output_image_list = [os.path.join(output_image_folder, image_name+'.png') for image_name in image_name_list]
    export_images(max_length, input_image_list, output_image_list, ouptut_denoise_list, input_mask_name_list)

    image_h, image_w = cv.imread(input_image_list[0]).shape[:2]
    scale = max_length / max(image_h, image_w)
    image_new_h, image_new_w = round(image_h * scale), round(image_w * scale)
    print(f'Image Shape : {image_h}x{image_w} -> {image_new_h}x{image_new_w}. Scale : {scale}')

    input_camera_npz_file = os.path.join(input_path, "cameras.npz")
    output_camera_npz_file = os.path.join(output_path, "cameras_sphere.npz")
    convert_camera_npz(input_camera_npz_file, output_camera_npz_file, scale)

    split_file = os.path.join(output_path, "split_sub1.pkl")
    export_traintest_split(split_file, [image_name+'.png' for image_name in image_name_list], div=1)
    split_file = os.path.join(output_path, "split_sub3.pkl")
    export_traintest_split(split_file, [image_name+'.png' for image_name in image_name_list], div=3)
    split_file = os.path.join(output_path, "split_sub5.pkl")
    export_traintest_split(split_file, [image_name+'.png' for image_name in image_name_list], div=5)
    split_file = os.path.join(output_path, "split_sub6.pkl")
    export_traintest_split(split_file, [image_name+'.png' for image_name in image_name_list], div=6)

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
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

if __name__ == "__main__":
    INPATH = './raw/DTU/scan24'
    OUTPATH = './processored/DTU/scan24'
    convert_data(INPATH, OUTPATH, 480)
    INPATH = './raw/DTU/scan37'
    OUTPATH = './processored/DTU/scan37'
    convert_data(INPATH, OUTPATH, 480)
    INPATH = './raw/DTU/scan40'
    OUTPATH = './processored/DTU/scan40'
    convert_data(INPATH, OUTPATH, 480)
    
    pass