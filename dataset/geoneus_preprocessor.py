case_list = [
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
]

model_list = [
    "Scannetpp_pinhole/0a7cc12c0e",
    "Scannetpp_pinhole/0a184cf634",
    "Scannetpp_pinhole/6ee2fc1070",
    "Scannetpp_pinhole/7b6477cb95",
    "Scannetpp_pinhole/56a0ec536c",

    "Scannetpp_pinhole/9460c8889d",
    "Scannetpp_pinhole/a08d9a2476",
    "Scannetpp_pinhole/e0abd740ba",
    "Scannetpp_pinhole/f8062cb7ce",


    "Scannetpp_pinhole/49a82360aa",
    "Scannetpp_pinhole/825d228aec",
    "Scannetpp_pinhole/6464461276",
    "Scannetpp_pinhole/c413b34238",
]

split_list = [
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
]

import os
import pickle
import colmap_read_model as read_model
import numpy as np

def geo_process(case, split, model_folder):
    folder = os.path.join('processored', case)
    cams = os.path.join('processored', case, 'cameras_sphere.npz')
    split_file = os.path.join('processored', case, split)
    model_folder = os.path.join('raw', model_folder, 'colmap')
    save_file = os.path.join('processored', case, 'points.pkl')
    print('Cams : ', cams)
    assert os.path.exists(cams)
    print('Folder : ', folder)
    assert os.path.exists(folder)
    print('Split File : ', split_file)
    assert os.path.exists(split_file)
    print('Model File : ', model_folder)
    assert os.path.exists(model_folder)

    with open(os.path.join(split_file), 'rb') as file:
        serialized_list = file.read()
        train_list, eval_list = pickle.loads(serialized_list)
    print('Train List : ', train_list)
    scale_mat = np.load(cams)['scale_mat_%s' % os.path.basename(train_list[0]).split('.')[0]].astype(np.float32)
    scale_mat = np.linalg.inv(scale_mat)
    print('Scale mat : ', scale_mat)

    camerasfile = os.path.join(model_folder, 'cameras.txt')
    camdata = read_model.read_cameras_text(camerasfile)
    
    imagesfile = os.path.join(model_folder, 'images.txt')
    imdata = read_model.read_images_text(imagesfile)

    points3dfile = os.path.join(model_folder, 'points3D.txt')
    pts3d = read_model.read_points3D_text(points3dfile)

    # for k in imdata:
    #     print(k, '<->', imdata[k].name)

    view_data = []

    names = [imdata[k].name for k in imdata]
    perm = np.argsort(names)
    for ori_idx in train_list:
        idx = int(ori_idx[:-4])
        pic_name = names[perm[idx]]
        print(ori_idx, ' <-> ', pic_name)
        for k in imdata:
            if imdata[k].name == pic_name:
                temp_points = []
                for ptsid in imdata[k].point3D_ids:
                    if ptsid != -1:
                        pointsxyz = pts3d[ptsid].xyz
                        pointsxyzw = np.array([*pointsxyz, 1])
                        pointsxyzw = pointsxyzw[:, None]
                        pointsxyzw_trans = scale_mat @ pointsxyzw
                        pointsnormalxyz = pointsxyzw_trans[:3, 0] / pointsxyzw_trans[3:4, 0]
                        pointsnormallength = np.linalg.norm(pointsnormalxyz)
                        # print(pointsnormalxyz, '<->', pointsnormallength)
                        if pointsnormallength <= 1:
                            temp_points.append(pts3d[ptsid].xyz)
                temp_array = np.array(temp_points)
                print('Points Shape : ', temp_array.shape)
                view_data.append(temp_array)
                break
    
    with open(save_file, 'wb') as f:
        pickle.dump(view_data, f)

if __name__ == "__main__":
    print('hello')
    for i in range(len(case_list)):
        geo_process(case_list[i], split_list[i], model_list[i])
        # exit()
