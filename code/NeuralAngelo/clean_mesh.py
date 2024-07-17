import numpy as np
import cv2 as cv
import os
from glob import glob
from scipy.io import loadmat
import trimesh
import open3d as o3d
import torch
from tqdm import tqdm

import sys

sys.path.append("../")

DATA_PATH = '/home/wxl/neuralangelo/public_data/dtu'
LOG_PATH = 'logs/dtu/'
CASE = 'dtu_scan24'
# case_list = ['baby', 'bear', 'bell', 'clock', 'deaf', 'farmer', 'pavilion', 'sculpture']
case_list = ['dtu_scan24']

def gen_rays_from_single_image(H, W, image, intrinsic, c2w, depth=None, mask=None):
    """
    generate rays in world space, for image image
    :param H:
    :param W:
    :param intrinsics: [3,3]
    :param c2ws: [4,4]
    :return:
    """
    device = image.device
    ys, xs = torch.meshgrid(torch.linspace(0, H - 1, H),
                            torch.linspace(0, W - 1, W))  # pytorch's meshgrid has indexing='ij'
    p = torch.stack([xs, ys, torch.ones_like(ys)], dim=-1)  # H, W, 3

    # normalized ndc uv coordinates, (-1, 1)
    ndc_u = 2 * xs / (W - 1) - 1
    ndc_v = 2 * ys / (H - 1) - 1
    rays_ndc_uv = torch.stack([ndc_u, ndc_v], dim=-1).view(-1, 2).float().to(device)

    intrinsic_inv = torch.inverse(intrinsic)

    p = p.view(-1, 3).float().to(device)  # N_rays, 3
    p = torch.matmul(intrinsic_inv[None, :3, :3], p[:, :, None]).squeeze()  # N_rays, 3
    rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # N_rays, 3
    rays_v = torch.matmul(c2w[None, :3, :3], rays_v[:, :, None]).squeeze()  # N_rays, 3
    rays_o = c2w[None, :3, 3].expand(rays_v.shape)  # N_rays, 3

    image = image.permute(1, 2, 0)
    color = image.view(-1, 1)
    depth = depth.view(-1, 1) if depth is not None else None
    mask = mask.view(-1, 1) if mask is not None else torch.ones([H * W, 1]).to(device)
    sample = {
        'rays_o': rays_o,
        'rays_v': rays_v,
        'rays_ndc_uv': rays_ndc_uv,
        'rays_color': color,
        # 'rays_depth': depth,
        'rays_mask': mask,
        'rays_norm_XYZ_cam': p  # - XYZ_cam, before multiply depth
    }
    if depth is not None:
        sample['rays_depth'] = depth

    return sample


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
    pose[:3, :3] = R.transpose()  # ? why need transpose here
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose  # ! return cam2world matrix here


def clean_points_by_mask(points, imgs_idx=None, minimal_vis=0, mask_dilated_size=11):
    cameras = np.load(os.path.join(DATA_PATH, CASE, 'cameras_sphere.npz'))
    image_lis = sorted(glob(os.path.join(DATA_PATH, CASE, 'image','*.png')))
    not_inside_mask = np.zeros(len(points))
    inside_mask = np.zeros(len(points))
    mask_lis = sorted(glob(os.path.join(DATA_PATH, CASE, 'mask','*.png')))
    mask_lis = [cv.imread(i) for i in mask_lis]
    print('Mask List 0 : ', mask_lis[0].shape)
    mask_lis = [i[:, :, -2:-1] for i in mask_lis]
    h, w = mask_lis[0].shape[0], mask_lis[0].shape[1]
    print('Mask List 0 : ', mask_lis[0].shape, ' [', mask_lis[0].min(), ',', mask_lis[0].max(), ']')
    cv.imwrite('test.png', (mask_lis[0]>128).astype(int)*255)

    if imgs_idx is None:
        imgs_idx = [str(os.path.basename(i)[:-4]) for i in image_lis]

    print(imgs_idx)

    # imgs_idx = [i for i in range(n_images)]
    for (li, i) in enumerate(imgs_idx):
        print(li, ' <-> ', i)
        # print('Scale : ', cameras['scale_mat_{}'.format(int(i))])
        world_mat = cameras['world_mat_{}'.format(int(i))]
        scale_mat = cameras['scale_mat_{}'.format(int(i))]
        P = world_mat @ scale_mat
        P = P[:3, :4]
        # print('Points', points.shape)
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32)
        # print('pts_image', pts_image.shape, '[', pts_image.min(), ',',pts_image.max() , ']')

        mask_image = mask_lis[li]
        # print(f'Mask List {li} : ', mask_lis[0].shape, ' [', mask_lis[0].min(), ',', mask_lis[0].max(), ']')
        kernel_size = mask_dilated_size  # default 101
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)
        mask_image = (mask_image[:, :] > 128)
        # mask_image = np.ones_like(mask_image).astype(bool)
        cv.imwrite('test_{}.png'.format(li), mask_image.astype(int)*255)
        # print(mask_image.shape)
        mask_image = np.concatenate([np.zeros([1, w]), mask_image, np.zeros([1, w])], axis=0)
        # print(mask_image.shape)
        mask_image = np.concatenate([np.zeros([h+2, 1]), mask_image, np.zeros([h+2, 1])], axis=1) # h+2 w+2
        # print(mask_image.shape)

        in_pic = ((pts_image[:, 0] >= 0) * (pts_image[:, 0] < w) * (pts_image[:, 1] >= 0) * (pts_image[:, 1] < h)) > 0
        curr_mask = mask_image[((pts_image[:, 1]+1).clip(0, h+1), (pts_image[:, 1]+1).clip(0, w+1))]
        not_in_mask = 1-curr_mask
        curr_not_in_mask = not_in_mask.astype(np.float32) * in_pic
        curr_in_mask = curr_mask.astype(np.float32) * in_pic
        print('Curr: ', curr_not_in_mask.min(), ',', curr_not_in_mask.max())
        print('Inside: ', curr_in_mask.min(), ',', curr_in_mask.max())
        inside_mask += curr_in_mask
        not_inside_mask += curr_not_in_mask
    
    print('Inside: ', curr_mask.min(), ',', curr_mask.max())
    return (not_inside_mask > 0) * (inside_mask < 0.1)


def clean_mesh_faces_by_mask(mesh_file, new_mesh_file, imgs_idx, minimal_vis=0, mask_dilated_size=11):
    old_mesh = trimesh.load(mesh_file)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask = clean_points_by_mask(old_vertices, imgs_idx, minimal_vis, mask_dilated_size)
    mask = ~mask
    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(int)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)

    new_mesh.export(new_mesh_file)


def clean_mesh_by_faces_num(mesh, faces_num=500):
    old_vertices = mesh.vertices[:]
    old_faces = mesh.faces[:]

    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=faces_num)
    mask = np.zeros(len(mesh.faces), dtype=np.bool)
    mask[np.concatenate(cc)] = True

    indexes = np.ones(len(old_vertices)) * -1
    indexes = indexes.astype(np.long)
    indexes[np.where(mask)] = np.arange(len(np.where(mask)[0]))

    faces_mask = mask[old_faces[:, 0]] & mask[old_faces[:, 1]] & mask[old_faces[:, 2]]
    new_faces = old_faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = old_vertices[np.where(mask)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)

    return new_mesh


def clean_mesh_faces_outside_frustum(old_mesh_file, new_mesh_file, imgs_idx, H=1200, W=1600, mask_dilated_size=11,
                                     isolated_face_num=500, keep_largest=True):
    '''Remove faces of mesh which cannot be orserved by all cameras
    '''
    # if path_mask_npz:
    #     path_save_clean = IOUtils.add_file_name_suffix(path_save_clean, '_mask')

    cameras = np.load(os.path.join(DATA_PATH, CASE, 'cameras_sphere.npz'))
    image_lis = sorted(glob(os.path.join(DATA_PATH, CASE, 'image','*.png')))
    mask_lis = [cv.imread(i) for i in image_lis]
    print('Mask List 0 : ', mask_lis[0].shape)
    mask_lis = [i[:, :, -1:] for i in mask_lis]
    print('Mask List 0 : ', mask_lis[0].shape, ' [', mask_lis[0].min(), ',', mask_lis[0].max(), ']')

    mesh = trimesh.load(old_mesh_file)
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

    if imgs_idx is None:
        imgs_idx = [str(os.path.basename(i)[:-4]) for i in image_lis]

    print(imgs_idx)
    
    all_indices = []
    chunk_size = 5120
    for li, i in enumerate(imgs_idx):
        mask_image = mask_lis[li]
        kernel_size = mask_dilated_size  # default 101
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_image = cv.dilate(mask_image, kernel, iterations=1)[:, :, None]

        world_mat = cameras['world_mat_{}'.format(i)]
        scale_mat = cameras['scale_mat_{}'.format(i)]
        P = world_mat @ scale_mat
        P = P[:3, :4]
        
        intrinsic, pose = load_K_Rt_from_P(None, P[:3, :])

        rays = gen_rays_from_single_image(H, W, torch.from_numpy(mask_image).permute(2, 0, 1).float(),
                                          torch.from_numpy(intrinsic)[:3, :3].float(),
                                          torch.from_numpy(pose).float())
        rays_o = rays['rays_o']
        rays_d = rays['rays_v']
        rays_mask = rays['rays_color']

        rays_o = rays_o.split(chunk_size)
        rays_d = rays_d.split(chunk_size)
        rays_mask = rays_mask.split(chunk_size)

        for rays_o_batch, rays_d_batch, rays_mask_batch in tqdm(zip(rays_o, rays_d, rays_mask)):
            rays_mask_batch = rays_mask_batch[:, 0] > 128
            rays_o_batch = rays_o_batch[rays_mask_batch]
            rays_d_batch = rays_d_batch[rays_mask_batch]

            idx_faces_hits = intersector.intersects_first(rays_o_batch.cpu().numpy(), rays_d_batch.cpu().numpy())
            all_indices.append(idx_faces_hits)

    values = np.unique(np.concatenate(all_indices, axis=0))
    mask_faces = np.ones(len(mesh.faces))
    mask_faces[values[1:]] = 0
    print(f'Surfaces/Kept: {len(mesh.faces)}/{len(values)}')

    mesh_o3d = o3d.io.read_triangle_mesh(old_mesh_file)
    print("removing triangles by mask")
    mesh_o3d.remove_triangles_by_mask(mask_faces)

    o3d.io.write_triangle_mesh(new_mesh_file, mesh_o3d)

    # # clean meshes
    new_mesh = trimesh.load(new_mesh_file)
    cc = trimesh.graph.connected_components(new_mesh.face_adjacency, min_len=500)
    mask = np.zeros(len(new_mesh.faces), dtype=bool)
    mask[np.concatenate(cc)] = True
    new_mesh.update_faces(mask)
    new_mesh.remove_unreferenced_vertices()
    new_mesh.export(new_mesh_file)

    # meshes = new_mesh.split(only_watertight=False)
    #
    # if not keep_largest:
    #     meshes = [mesh for mesh in meshes if len(mesh.faces) > isolated_face_num]
    #     # new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]
    #     merged_mesh = trimesh.util.concatenate(meshes)
    #     merged_mesh.export(new_mesh_file)
    # else:
    #     new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]
    #     new_mesh.export(new_mesh_file)

    o3d.io.write_triangle_mesh(new_mesh_file.replace(".ply", "_raw.ply"), mesh_o3d)
    print("finishing removing triangles")


def clean_outliers(old_mesh_file, new_mesh_file):
    new_mesh = trimesh.load(old_mesh_file)

    meshes = new_mesh.split(only_watertight=False)
    new_mesh = meshes[np.argmax([len(mesh.faces) for mesh in meshes])]

    new_mesh.export(new_mesh_file)


if __name__ == "__main__":

    mask_kernel_size = 11

    imgs_idx = None
    # imgs_idx = [42, 43, 44]
    # imgs_idx = [1, 8, 9]
    for j in case_list:
        CASE = j
        old_mesh_file = os.path.join(LOG_PATH, CASE, '150000.ply')
        clean_mesh_file = os.path.join(LOG_PATH, CASE, '150000_clean.ply')
        final_mesh_file = os.path.join(LOG_PATH, CASE, '150000_final.ply')

        clean_mesh_faces_by_mask(old_mesh_file, clean_mesh_file, imgs_idx, minimal_vis=48,
                                    mask_dilated_size=mask_kernel_size)
        # clean_mesh_faces_outside_frustum(clean_mesh_file, final_mesh_file, imgs_idx, mask_dilated_size=mask_kernel_size)
        # except:
        #     pass
    print("finish processing")