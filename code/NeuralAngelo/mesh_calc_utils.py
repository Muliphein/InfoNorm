import cv2 as cv
import torch
import trimesh
import os
from scipy.spatial import cKDTree as KDTree
import numpy as np
from tqdm import tqdm
import pyembree
import third_party.raytracing.raytracing as raytracing

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

def psnr_calc(list_gt, list_pred):
    psnr_list = []
    assert len(list_gt) == len(list_pred)
    for idx in range(len(list_gt)):
        gt_img = cv.imread(list_gt[idx]) / 256.0
        pred_img = cv.imread(list_pred[idx]) / 256.0
        psnr = -10 * (
            (torch.from_numpy(gt_img) - torch.from_numpy(pred_img)) ** 2
        ).mean().log10().item()
        psnr_list.append(psnr)
    return psnr_list

def trimesh_metrics(pred_mesh, gt_mesh, num_mesh_samples=500000, threshold_list = [0.001, 0.01, 0.05]):
    precision_list = []
    recall_list = []
    fscore_list = []
    pred_points_sampled = trimesh.sample.sample_surface(pred_mesh, num_mesh_samples)[0]
    gt_points_sampled = trimesh.sample.sample_surface(gt_mesh, num_mesh_samples)[0]
    gen_points_kd_tree = KDTree(pred_points_sampled)
    one_distances, one_vertex_ids = gen_points_kd_tree.query(gt_points_sampled)
    gt_points_kd_tree = KDTree(gt_points_sampled)
    two_distances, two_vertex_ids = gt_points_kd_tree.query(pred_points_sampled)

    for threshold in threshold_list:
        print(f"Calc Threshold: {threshold}")
        precision = (one_distances < threshold).astype(int).sum() / num_mesh_samples
        recall = (two_distances < threshold).astype(int).sum() / num_mesh_samples
        fscore = 2 * precision * recall / (precision + recall)
        precision_list.append(precision)
        recall_list.append(recall)
        fscore_list.append(fscore)
    gt_to_gen_chamfer = np.mean(np.square(one_distances))
    gen_to_gt_chamfer = np.mean(np.square(two_distances))

    return gt_to_gen_chamfer, gen_to_gt_chamfer, precision_list, recall_list, fscore_list

def mesh_calc(mesh_gt, mesh_pred, thres_list=[0.001, 0.005, 0.01, 0.05], scale=1.0, save_file=None):

    assert(save_file is not None)

    pred_mesh_name = mesh_pred
    gt_mesh_name = mesh_gt

    print(f'Pred {pred_mesh_name} <-> GT {gt_mesh_name}')
    
    import csv
    csv_file = save_file
    threshold_list = [i * scale for i in thres_list]
    print(f'Threshold List {threshold_list} Scale : {scale}')

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)

        pred_mesh = trimesh.load_mesh(pred_mesh_name)
        gt_mesh = trimesh.load_mesh(gt_mesh_name)
        
        gt2pred, pred2gen, precision, recall, fscore = trimesh_metrics(pred_mesh, gt_mesh, 50000, threshold_list)
        # gt2pred : precision
        # pred2gen : recall

        writer.writerow([' ', 'precision', 'recall', 'f-score'])
        writer.writerow(['Chamfer Distance(Lower is Better)', gt2pred, pred2gen])
        for idx, threshold in enumerate(thres_list):
            writer.writerow([f'Threshold {threshold} P&R&F(Higher is Better)', precision[idx], recall[idx], fscore[idx]])
        print('Mesh Calc Over')

def clean_points_by_mask(points, w2p_list, w, h):
    print(points.shape)
    inside_pic = np.zeros(len(points))

    for w2p in w2p_list:
        P = w2p[:3, :4]
        pts_image = np.matmul(P[None, :3, :3], points[:, :, None]).squeeze() + P[None, :3, 3]
        is_front = (pts_image[:, 2] > 0)
        # print(f'Pts Image [{pts_image.min()}, {pts_image.max()}]')
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32)

        in_pic = (
            (pts_image[:, 0] >= 0) *
            (pts_image[:, 0] < w) *
            (pts_image[:, 1] >= 0) *
            (pts_image[:, 1] < h) *
            is_front
        ) > 0
        inside_pic += in_pic.astype(np.float32)
    
    #! return the points need to be deleted, not in any pic 
    return (inside_pic == 0)

def clean_mesh_by_in_pic(mesh_file, new_mesh_file, w2p_list, w, h):
    old_mesh = trimesh.load(mesh_file)
    old_vertices = old_mesh.vertices[:]
    old_faces = old_mesh.faces[:]
    mask = clean_points_by_mask(old_vertices, w2p_list, w, h)
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

def clean_mesh_by_camera(input_file, output_file, w2p_list, w, h):
    input_mesh = trimesh.load(input_file)
    vertices = input_mesh.vertices
    faces = input_mesh.faces
    # print('Vectices : ', vertices.shape)
    # print('Faces : ', faces.shape)
    tbar = tqdm(total=len(w2p_list))
    visible_points = np.zeros_like(vertices[:, 0])
    for w2p in w2p_list:
        w2p = w2p[:3, :4]

        pts_image = np.matmul(w2p[None, :3, :3], vertices.copy()[:, :, None]).squeeze() + w2p[None, :3, 3]
        is_front = (pts_image[:, 2] > 0)
        pts_image = pts_image / pts_image[:, 2:]
        pts_image = np.round(pts_image).astype(np.int32)

        in_pic = (
            (pts_image[:, 0] >= 0) *
            (pts_image[:, 0] < w) *
            (pts_image[:, 1] >= 0) *
            (pts_image[:, 1] < h) *
            is_front
        ) > 0
        point_can_see = in_pic.astype(np.float32)

        c2p, c2w = load_K_Rt_from_P(None, w2p)
        origin = c2w[:3, 3]
        origins = np.tile(origin, (vertices.shape[0], 1))
        vertices = vertices.copy()
        directions = vertices - origins
        distance = np.linalg.norm(directions, axis=-1, keepdims=True)
        current_depth = (np.ones_like(vertices[:, 0]) * 1e6).astype(float) # For all points set far enough
        directions = directions / distance

        # !Use trimesh + pyembree Have Precision Issues
        # refer to https://github.com/mikedh/trimesh/issues/242

        # points, index_ray, index_tri = input_mesh.ray.intersects_location(
        #     origins, directions, multiple_hits=False
        # )
        # depth = trimesh.util.diagonal_dot(points - origins[0], directions[index_ray])
        # current_depth[index_ray] = depth

        # !Use raytracing to generate Precision result
        # !Make serveral times to absorb small components
        current_unvisible_points = np.zeros_like(vertices[:, 0]).astype(float)
        eps = 1e-3
        perturb = np.array([
            [0, 0, 0],
            # [eps, eps, eps],
            # [eps, eps, -eps],
            # [eps, -eps, eps],
            # [eps, -eps, -eps],
            # [-eps, eps, eps],
            # [-eps, eps, -eps],
            # [-eps, -eps, eps],
            # [-eps, -eps, -eps],
        ]).astype(float)
        old_vertices = vertices
        for i in range(len(perturb)):
            vertices = old_vertices.copy() # [N, 3]
            vertices = vertices + perturb[i][None,:]
            directions = vertices - origins
            distance = np.linalg.norm(directions, axis=-1, keepdims=True)
            directions = directions / distance
            RT = raytracing.RayTracer(input_mesh.vertices, input_mesh.faces) # build with numpy.ndarray
            origins_torch = torch.from_numpy(origins).cuda()
            directions_torch = torch.from_numpy(directions).cuda()
            intersections, face_normals, depth = RT.trace(origins_torch, directions_torch) # [N, 3], [N, 3], [N,]
            current_depth = depth.detach().cpu().numpy()
            # print(f'Depth Info: [{current_depth.min()},{current_depth.max()}]')
            current_unvisible_points += ((current_depth + 1e-3 + eps * 1.732) < distance[:, 0]).astype(float)



        # Cull the point after mesh
        #? In Picture and can be seen
        visible_points += (1-(current_unvisible_points>0.5).astype(float)) * point_can_see
        tbar.update(1)

    visible_points = visible_points > 0.5

    indexes = np.ones(len(vertices)) * -1
    indexes = indexes.astype(int)
    indexes[np.where(visible_points)] = np.arange(len(np.where(visible_points)[0]))

    faces_mask = visible_points[faces[:, 0]] & visible_points[faces[:, 1]] & visible_points[faces[:, 2]]
    new_faces = faces[np.where(faces_mask)]
    new_faces[:, 0] = indexes[new_faces[:, 0]]
    new_faces[:, 1] = indexes[new_faces[:, 1]]
    new_faces[:, 2] = indexes[new_faces[:, 2]]
    new_vertices = vertices[np.where(visible_points)]

    new_mesh = trimesh.Trimesh(new_vertices, new_faces)

    new_mesh.export(output_file)

def get_w2p_from_camfile(cam_file, name_lis):
    cam_dict = np.load(cam_file)
    w2p_list = [cam_dict['world_mat_'+name] for name in name_lis]
    return w2p_list

def get_scale_from_camfile(cam_file, name):
    cam_dict = np.load(cam_file)
    scale_mat = cam_dict['scale_mat_'+name]
    assert(scale_mat[0][0] == scale_mat[1][1])
    assert(scale_mat[0][0] == scale_mat[2][2])
    # three dimension scale should be the same
    return scale_mat[0][0]

def subdivide_mesh(input_file, output_file, max_length):
    mesh = trimesh.load_mesh(input_file)
    subdivided_mesh = mesh.subdivide_to_size(max_edge=max_length, max_iter=100)
    subdivided_mesh.export(output_file)

def clean_mesh(input_mesh, mid_mesh, output_mesh, cam_file, name_lis, w, h, extract_length = 512):

    w2p_list = get_w2p_from_camfile(cam_file, name_lis)

    #* Step O: subdivide the mesh for better clean and raytracing
    #! Roman numerals donnot contain zero !!!!
    scale = get_scale_from_camfile(cam_file, name_lis[0])
    max_length = 1.0 / extract_length * scale * 2.0
    # extract length = the extract resolution
    subdivide_mesh(input_mesh, mid_mesh, max_length)
    print('Sub Divide Over')

    #* Step I: find the points in any pic, others deleted -- Skiped
    # clean_mesh_by_in_pic(mid_mesh, mid_mesh, w2p_list, w, h)

    w2p_list = get_w2p_from_camfile(cam_file, name_lis)

    #* Step II: find the points in any camera can see, others deleted
    clean_mesh_by_camera(mid_mesh, output_mesh, w2p_list, w, h)

    pass

def convert_mesh(input_mesh, output_mesh, trans_file=None, scale_mat=None, inv=False):
    input_mesh = trimesh.load(input_mesh)
    vertices = input_mesh.vertices
    faces = input_mesh.faces

    assert (trans_file is None and scale_mat is not None) or (trans_file is not None and scale_mat is None)

    if trans_file is not None:
        try:
            scale_mat = np.loadtxt(trans_file)
        except:
            scale_mat = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

    if inv:
        scale_mat = np.linalg.inv(scale_mat)

    print(f'Apply Scale mat : {scale_mat}')

    vertices = np.concatenate([vertices, np.ones_like(vertices[:, :1])], axis=1)

    vertices = vertices.transpose()
    vertices = scale_mat @ vertices
    vertices = vertices[:3, :].transpose()
    print(vertices.shape)

    new_mesh = trimesh.Trimesh(vertices, faces)
    
    new_mesh.export(output_mesh)

    pass 