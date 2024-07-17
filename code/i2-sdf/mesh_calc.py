import os
import pickle
import numpy as np
from mesh_calc_utils import mesh_calc, psnr_calc, clean_mesh, convert_mesh
import cv2 as cv
from glob import glob

def mesh_calc_case(case_folder, data_root ,recalc=False):
    case_folder_ori = case_folder
    mesh_file = os.path.join(case_folder, 'eval', 'mesh', 'result.ply')
    print('Mesh Name : ', mesh_file)
    case_folder = os.path.dirname(case_folder)
    print('Folder : ', case_folder)

    base_name = str(case_folder)

    split_lists = base_name.split('/')
    for i, split in enumerate(split_lists):
        if 'split_' in split:
            split_name = split
            break
    case_name = ''
    for j in range(i+1, len(split_lists)):
        case_name = case_name + '/' + split_lists[j]
    case_name = case_name[2:]
    print('Split name : ', split_name)
    print('Case name : ', case_name)
    case_folder = case_folder_ori

    data_folder = os.path.join(data_root, case_name)
    print(data_folder)
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
    logs_paths = glob(os.path.join(logs_root, '*', '*', '*', '*', '*', 'eval', 'mesh', 'result.ply'))
    # print(logs_paths)
    for case in logs_paths:
        for i in range(3):
            case = os.path.dirname(case)
        # print(case)
        mesh_calc_case(case, data_root, recalc=recalc)
        # exit()