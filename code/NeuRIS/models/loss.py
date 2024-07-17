import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from itertools import combinations
import math
import utils.utils_training as TrainingUtils

MIN_PIXELS_PLANE = 20

def get_normal_consistency_loss(normals, mask_curr_plane, error_mode = 'angle_error'):
    '''Return normal loss of pixels on the same plane

    Return:
        normal_consistency_loss: float, on each pixels
        num_pixels_curr_plane: int
        normal_mean_curr, 3*1
    '''
    num_pixels_curr_plane = mask_curr_plane.sum()
    if num_pixels_curr_plane < MIN_PIXELS_PLANE:
        return 0.0,  num_pixels_curr_plane, torch.zeros(3)

    normals_fine_curr_plane = normals * mask_curr_plane
    normal_mean_curr = normals_fine_curr_plane.sum(dim=0) / num_pixels_curr_plane

    if error_mode == 'angle_error':
        inner = (normals * normal_mean_curr).sum(dim=-1,keepdim=True)
        norm_all =  torch.linalg.norm(normals, dim=-1, ord=2,keepdim=True)
        norm_mean_curr = torch.linalg.norm(normal_mean_curr, dim=-1, ord=2,keepdim=True)
        angles = torch.arccos(inner/((norm_all*norm_mean_curr) + 1e-6)) #.clip(-np.pi, np.pi)
        angles = angles*mask_curr_plane
        normal_consistency_loss = F.l1_loss(angles, torch.zeros_like(angles), reduction='sum')
    
    return normal_consistency_loss, num_pixels_curr_plane, normal_mean_curr

def get_plane_offset_loss(pts, ave_normal, mask_curr_plane, mask_subplanes):
    '''
    Args:
        pts: pts in world coordinates
        normals: normals of pts in world coordinates
        mask_plane: mask of pts which belong to the same plane
    '''
    mask_subplanes_curr = copy.deepcopy(mask_subplanes)
    mask_subplanes_curr[mask_curr_plane == False] = 0 # only keep subplanes of current plane
    
    loss_offset_subplanes = []
    num_subplanes = int(mask_subplanes_curr.max().item())
    if num_subplanes < 1:
        return 0, 0
    
    num_pixels_valid_subplanes = 0
    loss_offset_subplanes = torch.zeros(num_subplanes)
    for i in range(num_subplanes):
        curr_subplane = (mask_subplanes_curr == (i+1))
        num_pixels = curr_subplane.sum()
        if num_pixels < MIN_PIXELS_PLANE:
            continue
        
        offsets = (pts*ave_normal).sum(dim=-1,keepdim=True)
        ave_offset = ((offsets * curr_subplane).sum() / num_pixels) #.detach()  # detach?

        diff_offsset = (offsets-ave_offset)*curr_subplane
        loss_tmp = F.mse_loss(diff_offsset, torch.zeros_like(diff_offsset), reduction='sum') #/ num_pixels

        loss_offset_subplanes[i] = loss_tmp
        num_pixels_valid_subplanes += num_pixels
    
    return loss_offset_subplanes.sum(), num_pixels_valid_subplanes

def get_manhattan_normal_loss(normal_planes):
    '''The major planes should be vertical to each other
    '''
    normal_planes = torch.stack(normal_planes, dim=0)
    num_planes = len(normal_planes)
    assert num_planes < 4
    if num_planes < 2:
        return 0

    all_perms = np.array( list(combinations(np.arange(num_planes),2)) ).transpose().astype(int) # 2*N
    normal1, normal2 = normal_planes[all_perms[0]], normal_planes[all_perms[1]]
    inner = (normal1 * normal2).sum(-1)
    manhattan_normal_loss = F.l1_loss(inner, torch.zeros_like(inner), reduction='mean')
    return manhattan_normal_loss

class NeuSLoss(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.color_weight = conf['color_weight']

        self.igr_weight = conf['igr_weight']
        self.smooth_weight = conf['smooth_weight']
        self.mask_weight = conf['mask_weight']

        self.depth_weight = conf['depth_weight']
        self.normal_weight = conf['normal_weight']

        self.normal_consistency_weight = conf['normal_consistency_weight']
        self.plane_offset_weight = conf['plane_offset_weight']
        self.manhattan_constrain_weight = conf['manhattan_constrain_weight']
        self.plane_loss_milestone = conf['plane_loss_milestone']

        self.warm_up_start = conf['warm_up_start']
        self.warm_up_end = conf['warm_up_end']

        self.jacobi_contrastive_weight = conf['jacobi_contrastive_weight']
        self.jacobi_gradient_weight = conf['jacobi_gradient_weight']
        self.jacobi_pos_threshold = conf['jacobi_pos_threshold']

        self.jacobi_start = conf['jacobi_start']
        self.jacobi_end = conf['jacobi_end']
        self.jacobi_type = conf.get_string('jacobi_type', default='vector')
        self.no_abs = conf.get_bool('no_abs', default=False)
        self.normal_gather = conf.get_bool('normal_gather', default=False)
        print(f'[notice] loss normalgather')
        print(f'Jacobi Type : {self.jacobi_type}')

        self.iter_step = 0
        self.iter_end = -1

    @staticmethod
    def torch_space_differ(A): # bs, n, l
        # A is normalized
        bs, n, l = A.shape

        A = [A[:, i, :] for i in range(3)]

        for i in range(3):
            for j in range(i):
                alpha = (A[i] * A[j]).sum(dim=-1)
                project = (alpha / ((A[j]).square().sum(dim=-1)+1e-2))[:, None] * A[j]
                A[i] = A[i] - project
        space_size = (1+1) * (torch.linalg.norm(A[1], dim=-1)+1) * (torch.linalg.norm(A[2], dim=-1)+1) # bs

        # Progress 2 ortho div
        A = torch.stack(A, dim=1)
        B = [A[None, :, i:i+1, :] for i in range(3)]
        A = A[:, None,:, :].expand((bs, bs, n, l))

        for i in range(n):
            alpha = (B[i] * A).sum(dim=-1)
            project = (alpha / (B[i] .square().sum(dim=-1)+1e-2))[:, :, :, None] * B[i]
            A = A - project
        SpaceA = A

        # Progress 3 orhtho again
        A = [A[:, :, i, :] for i in range(3)]
        for i in range(n):
            for j in range(i):
                alpha = (A[i] * A[j]).sum(dim=-1) # bs, bs
                project = (alpha / (A[j].square().sum(dim=-1)+1e-2))[:, :, None] * A[j] # projection A_i on A_j
                A[i] = A[i] - project # delete the projection
        SpaceA = torch.stack(A, dim=2)

        space_div_length = torch.linalg.norm(SpaceA, dim=-1) # bs, bs, n
        space_div_size = torch.prod(space_div_length+1, dim=-1) # bs, bs
        space_diff = space_size[:, None] / space_div_size
        return space_diff

    def jacobi_loss(self, pos_condition, feature, jaco_mask):
        # print('Jaco Shape : ', dino.shape)
        # print('Feature Shape : ', feature.shape)
        
        # Dino Space
        mask = jaco_mask
        
        if self.jacobi_type == 'vector':
            # Vector MI Calc
            feature = feature.reshape(feature.shape[0], -1)
            feature_dot = (feature @ torch.transpose(feature, 0, 1))
            feature_norm = torch.linalg.norm(feature, dim=-1)
            feature_norm_square = feature_norm * (feature_norm.unsqueeze(1)) + 1e-7
            if self.no_abs:
                # print('No ABS')
                mi_square = torch.exp(feature_dot / feature_norm_square)
            else:
                mi_square = torch.exp(torch.abs(feature_dot / feature_norm_square))
            
        elif self.jacobi_type == 'space':
            # Space MI Calc
            feature_norm = torch.linalg.norm(feature, dim=-1) # [bs, 3, 256] -> [bs, 3]
            feature_normed = feature / feature_norm[:, :, None]
            mi_square = self.torch_space_differ(feature_normed) # [bs, 3, 256] -> [bs, bs]
        
        else: raise NotImplementedError 
        
        # Sum Results
        pos_sim_total = (pos_condition.float() * mi_square).sum(dim=-1)
        neg_sim_total = ((~pos_condition).float() * mi_square).sum(dim=-1)
        contrastive = torch.log(pos_sim_total/(pos_sim_total + neg_sim_total) + 1e-7)

        # Loss Calculate
        jacobi_gradient_loss = torch.mean((feature_norm - 1.0) ** 2)
        contrastive_no_nan_count = torch.sum(~torch.isnan(contrastive))
        mi_contrastive_loss = -torch.nansum(contrastive * mask) / contrastive_no_nan_count
        
        if self.normal_gather:
            return jacobi_gradient_loss, torch.nanmean(torch.log(pos_sim_total + 0.1))
        
        return jacobi_gradient_loss, mi_contrastive_loss

    def get_warm_up_ratio(self):
        if self.warm_up_end == 0.0:
            return 1.0
        elif self.iter_step < self.warm_up_start:
            return 0.0
        else:
            return np.min([1.0, (self.iter_step - self.warm_up_start) / (self.warm_up_end - self.warm_up_start)])

    def forward(self, input_model, render_out, sdf_network_fine, patchmatch_out = None):
        true_rgb = input_model['true_rgb']
        jaco_info = input_model['jaco_info']

        mask, rays_o, rays_d, near, far = input_model['mask'], input_model['rays_o'], input_model['rays_d'],  \
                                                    input_model['near'], input_model['far']
        mask_sum = mask.sum() + 1e-5
        batch_size = len(rays_o)

        color_fine = render_out['color_fine']
        variance = render_out['variance']
        cdf_fine = render_out['cdf_fine']
        gradient_error_fine = render_out['gradient_error_fine']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        weights = render_out['weights']
        depth = render_out['depth']
        feature = render_out['feature']
        jaco_mask = input_model['jaco_mask']

        planes_gt = None
        if 'planes_gt' in input_model:
            planes_gt = input_model['planes_gt']
        if 'subplanes_gt' in input_model:
            subplanes_gt = input_model['subplanes_gt']
        
        logs_summary = {}

        # patchmatch loss
        normals_target, mask_use_normals_target = None, None
        pts_target, mask_use_pts_target = None, None
        if patchmatch_out is not None:
            if patchmatch_out['patchmatch_mode'] == 'use_geocheck':
                mask_use_normals_target = (patchmatch_out['idx_scores_min'] > 0).float()
                normals_target = input_model['normals_gt']
            else:
                raise NotImplementedError
        else:
            if self.normal_weight>0:
                normals_target = input_model['normals_gt']
                mask_use_normals_target = torch.ones(batch_size, 1).bool()

        # Color loss
        color_fine_loss, background_loss, psnr = 0, 0, 0
        if True:
            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
                
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            # Mask loss, optional
            background_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            logs_summary.update({           
                'Loss/loss_color':  color_fine_loss.detach().cpu(),
                'Loss/loss_bg':     background_loss
            })
        
        # Eikonal loss
        gradient_error_loss = 0
        if self.igr_weight > 0:
            gradient_error_loss = gradient_error_fine
            logs_summary.update({           
                'Loss/loss_eik':    gradient_error_loss.detach().cpu(),
            })            
        
        # Smooth loss, optional
        surf_reg_loss = 0.0
        if self.smooth_weight > 0:
            depth = render_out['depth'].detach()
            pts = rays_o + depth * rays_d
            n_pts = pts + torch.randn_like(pts) * 1e-3  # WARN: Hard coding here
            surf_normal = sdf_network_fine.gradient(torch.cat([pts, n_pts], dim=0)).squeeze()
            surf_normal = surf_normal / torch.linalg.norm(surf_normal, dim=-1, ord=2, keepdim=True)

            surf_reg_loss_pts = (torch.linalg.norm(surf_normal[:batch_size, :] - surf_normal[batch_size:, :], ord=2, dim=-1, keepdim=True))
            # surf_reg_loss = (surf_reg_loss_pts*pixel_weight).mean()
            surf_reg_loss = surf_reg_loss_pts.mean()

        # normal loss
        normals_fine_loss, mask_keep_gt_normal = 0.0, torch.ones(batch_size)
        if self.normal_weight > 0 and normals_target is not None:
            normals_gt = normals_target # input_model['normals_gt'] #
            normals_fine = render_out['normal']
            
            normal_certain_weight = torch.ones(batch_size, 1).bool()
            if 'normal_certain_weight' in input_model:
                normal_certain_weight = input_model['normal_certain_weight']

            thres_clip_angle = -1 #
            normal_certain_weight = normal_certain_weight*mask_use_normals_target
            angular_error, mask_keep_gt_normal = TrainingUtils.get_angular_error(normals_fine, normals_gt, normal_certain_weight, thres_clip_angle)

            normals_fine_loss = angular_error
            logs_summary.update({
                'Loss/loss_normal_gt': normals_fine_loss
            })

        # depth loss, optional
        depths_fine_loss = 0.0
        if self.depth_weight > 0 and (pts_target is not None):
            pts = rays_o + depth * rays_d
            pts_error = (pts_target - pts) * mask_use_pts_target
            pts_error = torch.linalg.norm(pts_error, dim=-1, keepdims=True)

            depths_fine_loss = F.l1_loss(pts_error, torch.zeros_like(pts_error), reduction='sum') / (mask_use_pts_target.sum()+1e-6)
            logs_summary.update({
                'Loss/loss_depth': depths_fine_loss,
                'Log/num_depth_target_use': mask_use_pts_target.sum().detach().cpu()
            })

        jacobi_loss = 0.0
        jacobi_gradient_loss = 0.0
        jacobi_contrastive_loss = 0.0
        if self.jacobi_contrastive_weight > 0 or self.jacobi_gradient_weight > 0:
            if self.iter_step >= self.jacobi_start and self.iter_step < self.jacobi_end:
                assert (jaco_info is not None), "Jaco Info Required"
                assert (feature is not None), "Render Feature Required"
                jacobi_gradient_loss, jacobi_contrastive_loss = self.jacobi_loss(jaco_info, feature, jaco_mask)
                jacobi_gradient_loss = self.jacobi_gradient_weight * jacobi_gradient_loss
                jacobi_contrastive_loss = self.jacobi_contrastive_weight * jacobi_contrastive_loss
                jacobi_loss = jacobi_contrastive_loss + jacobi_gradient_loss
            else:
                pass
                # print(f'Range {self.jacobi_start}, {self.jacobi_end} Now {self.iter_step}')
        logs_summary.update({
            'Loss/jacobi_loss': jacobi_loss,
            'Log/jacobi_contrastive_loss': jacobi_contrastive_loss,
            'Log/jacobi_gradient_loss': jacobi_gradient_loss,
        })

        plane_loss_all = 0
        if self.normal_consistency_weight > 0 and self.iter_step > self.plane_loss_milestone:
            num_planes = int(planes_gt.max().item())

            depth = render_out['depth']   # detach?
            pts = rays_o + depth * rays_d
            normals_fine = render_out['normal']

            # (1) normal consistency loss
            num_pixels_on_planes = 0
            num_pixels_on_subplanes = 0

            dominant_normal_planes = []
            normal_consistency_loss = torch.zeros(num_planes)
            loss_plane_offset = torch.zeros(num_planes)
            for i in range(int(num_planes)):
                idx_plane = i + 1
                mask_curr_plane = planes_gt.eq(idx_plane)
                if mask_curr_plane.float().max() < 1.0:
                    # this plane is not existent
                    continue
                consistency_loss_tmp, num_pixels_tmp, normal_mean_curr = get_normal_consistency_loss(normals_fine, mask_curr_plane)
                normal_consistency_loss[i] = consistency_loss_tmp
                num_pixels_on_planes += num_pixels_tmp

                # for Manhattan loss
                if i < 3:
                    # only use the 3 dominant planes
                    dominant_normal_planes.append(normal_mean_curr)
                
                # (2) plane-to-origin offset loss
                if self.plane_offset_weight > 0:
                    # normal_mean_curr_no_grad =  normal_mean_curr.detach()
                    plane_offset_loss_curr, num_pixels_subplanes_valid_curr = get_plane_offset_loss(pts, normal_mean_curr, mask_curr_plane, subplanes_gt)
                    loss_plane_offset[i] = plane_offset_loss_curr
                    num_pixels_on_subplanes += num_pixels_subplanes_valid_curr
            
            assert num_pixels_on_planes >= MIN_PIXELS_PLANE                   
            normal_consistency_loss = normal_consistency_loss.sum() / (num_pixels_on_planes+1e-6)
            loss_plane_offset = loss_plane_offset.sum() / (num_pixels_on_subplanes+1e-6)
            
            # (3) normal manhattan loss
            loss_normal_manhattan = 0
            if self.manhattan_constrain_weight > 0:
                loss_normal_manhattan = get_manhattan_normal_loss(dominant_normal_planes)

            plane_loss_all = normal_consistency_loss * self.normal_consistency_weight  + \
                                    loss_normal_manhattan * self.manhattan_constrain_weight + \
                                    loss_plane_offset * self.plane_offset_weight

        loss = color_fine_loss * self.color_weight +\
                gradient_error_loss * self.igr_weight +\
                surf_reg_loss * self.smooth_weight +\
                plane_loss_all +\
                background_loss * self.mask_weight +\
                normals_fine_loss * self.normal_weight * self.get_warm_up_ratio()  + \
                depths_fine_loss * self.depth_weight +\
                jacobi_loss

        logs_summary.update({
            'Loss/loss': loss.detach().cpu(),
            'Loss/loss_smooth': surf_reg_loss,
            'Loss/variance':    variance.mean().detach(),
            'Log/psnr':         psnr,
            'Log/ratio_warmup_loss':  self.get_warm_up_ratio(),
            'Loss/final_normal': normals_fine_loss * self.normal_weight * self.get_warm_up_ratio(),
            'Loss/final_eikonal': gradient_error_loss * self.igr_weight
        })
        return loss, logs_summary, mask_keep_gt_normal