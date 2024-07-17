import os
import pickle
import numpy as np
from mesh_calc_utils import mesh_calc, psnr_calc, clean_mesh, convert_mesh
import cv2 as cv
from glob import glob


case_list=(
    "Scannetpp_crop_scaled/0a7cc12c0e",

    "Scannetpp_crop_scaled/0a184cf634",
    "Scannetpp_crop_scaled/6ee2fc1070",
    "Scannetpp_crop_scaled/7b6477cb95",
    "Scannetpp_crop_scaled/56a0ec536c",

    "Scannetpp_crop_scaled/9460c8889d",
    "Scannetpp_crop_scaled/a08d9a2476",
    "Scannetpp_crop_scaled/e0abd740ba",
    "Scannetpp_crop_scaled/f8062cb7ce",


    "Scannetpp_crop_scaled/49a82360aa",
    "Scannetpp_crop_scaled/825d228aec",
    "Scannetpp_crop_scaled/6464461276",
    "Scannetpp_crop_scaled/c413b34238",
)

split_list=(
    "split_sub5.pkl",
    
    "split_sub5.pkl",
    "split_sub20.pkl",
    "split_sub5.pkl",
    "split_sub3.pkl",
    
    "split_sub10.pkl",
    "split_sub10.pkl",
    "split_sub5.pkl",
    "split_sub5.pkl",
    
    "split_sub5.pkl",
    "split_sub10.pkl",
    "split_sub10.pkl",
    "split_sub5.pkl",
)

def mesh_calc_case(case_folder, data_root ,recalc=False):
    case_folder_ori = case_folder
    mesh_file = os.path.join(case_folder, 'result.ply')
    print('Mesh Name : ', mesh_file)
    case_folder = os.path.dirname(case_folder)
    print('Folder : ', case_folder)

    base_name = str(case_folder)

    split_lists = base_name.split('/')
    case_name = split_lists[-3] + '/' + split_lists[-2]
    for split_item, case_item in zip(split_list, case_list):
        if case_item == case_name:
            split_name = split_item
            break
    print('Split name : ', split_name)
    print('Case name : ', case_name)
    case_folder = os.path.dirname(case_folder_ori)

    data_folder = os.path.join(data_root, case_name)
    print(data_folder)
    print('Check : ', os.path.exists(mesh_file), os.path.exists(data_folder))
    if os.path.exists(mesh_file) and os.path.exists(data_folder):
        if os.path.exists(os.path.join(case_folder, 'MESH.csv')):
            print(f'Skip Case Folder : {case_folder}')
            return
        print(f'Calc Case Folder : {case_folder}')
        gt_mesh_file = os.path.join(data_folder, 'mesh.ply')
        result_mesh_file = mesh_file
        cameras_name = os.path.join(data_folder, 'cameras_sphere.npz')
        split_file = os.path.join(data_folder, split_name)
        trans_file = os.path.join(data_folder, 'trans_n2w.txt')

        with open(split_file, 'rb') as file:
            serialized_list = file.read()
            train_list, eval_list = pickle.loads(serialized_list)

        scale_mat = np.load(cameras_name)[f"scale_mat_{str(train_list[0]).split('.')[0]}"]
        pic0_file = os.path.join(data_folder, 'image', train_list[0])
        img0 = cv.imread(pic0_file)
        H, W = img0.shape[0], img0.shape[1]

        if os.path.exists(trans_file):
            convert_mesh(
                input_mesh = gt_mesh_file,
                output_mesh = os.path.join(case_folder, 'gt-mesh.ply'),
                trans_file = trans_file,
                inv=True
            )
        else:
            convert_mesh(
                input_mesh = gt_mesh_file,
                output_mesh = os.path.join(case_folder, 'gt-mesh.ply'),
                scale_mat=scale_mat,
                inv=True
            )
        
        convert_mesh(
            input_mesh = os.path.join(case_folder, 'gt-mesh.ply'),
            output_mesh = os.path.join(case_folder, 'gt-mesh.ply'),
            scale_mat=scale_mat,
            inv=False
        )
        
        convert_mesh(
            input_mesh = result_mesh_file,
            output_mesh = os.path.join(case_folder, 'result.ply'),
            scale_mat=scale_mat,
            inv=False
        )
        result_mesh_file = os.path.join(case_folder, 'result.ply')

        clean_mesh(
            input_mesh=result_mesh_file,
            mid_mesh=os.path.join(case_folder, 'result-clean.ply'),
            output_mesh=os.path.join(case_folder, 'result-final.ply'),
            cam_file=cameras_name,
            name_lis=[name.split('.')[0] for name in train_list],
            w = W,
            h = H,
            extract_length = 512,
        )

        clean_mesh(
            input_mesh=os.path.join(case_folder, 'gt-mesh.ply'),
            mid_mesh=os.path.join(case_folder, 'gt-mesh-clean.ply'),
            output_mesh=os.path.join(case_folder, 'gt-mesh-final.ply'),
            cam_file=cameras_name,
            name_lis=[name.split('.')[0] for name in train_list],
            w = W,
            h = H,
            extract_length = 512,
        )

        mesh_calc(
            mesh_gt=os.path.join(case_folder, 'gt-mesh-final.ply'),
            mesh_pred=os.path.join(case_folder, 'result-final.ply'),
            thres_list=[0.001, 0.002, 0.005, 0.008, 0.01, 0.02, 0.03, 0.05],
            scale = scale_mat[0][0],
            save_file=os.path.join(case_folder, 'MESH.csv')
        )

if __name__ == "__main__":
    print('mesh calc')
    recalc = False
    logs_root = os.path.join('exps')
    data_root = os.path.join('..', '..', 'dataset', 'processored')
    logs_paths = glob(os.path.join(logs_root, '*', '*', '*', '*', '*', 'result.ply'))
    print(logs_paths)
    for case in logs_paths:
        for i in range(1):
            case = os.path.dirname(case)
        mesh_calc_case(case, data_root, recalc=recalc)
        # exit()