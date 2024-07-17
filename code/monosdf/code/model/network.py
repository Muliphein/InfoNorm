import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import matplotlib.pyplot as plt
import numpy as np

from torch import Tensor
class MyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.in_feature = in_features
        self.out_features = out_features

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    def batch_forward(self, input: Tensor) -> Tensor:
        input_shape = input.shape
        input = input.reshape(-1, input_shape[-1])
        bs = input.shape[0]
        self.weight_matrix = self.weight[None, :, :].expand(bs, self.out_features, self.in_features)
        output = torch.add(torch.einsum('ab,abc->ac', input, self.weight_matrix.transpose(1, 2)), self.bias)
        output = output.reshape(*input_shape[:-1], self.out_features)
        return output

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            sdf_bounding_sphere,
            d_in,
            d_out, # SDF
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
            inside_outside=False,
            with_dn_dw=False,
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [feature_vector_size]

        self.with_dn_dw = with_dn_dw
        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        print(multires, dims)
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)

                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if self.with_dn_dw:
            self.sdf_linear = MyLinear(dims[-2], d_out)
        else:
            self.sdf_linear = nn.Linear(dims[-2], d_out)

        if not inside_outside:
            torch.nn.init.normal_(self.sdf_linear.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            torch.nn.init.constant_(self.sdf_linear.bias, -bias)
        else:
            torch.nn.init.normal_(self.sdf_linear.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
            torch.nn.init.constant_(self.sdf_linear.bias, bias)

        if weight_norm:
            self.sdf_linear = nn.utils.weight_norm(self.sdf_linear)
            
        self.softplus = nn.Softplus(beta=100)
        

    def forward(self, input, use_batch_forward=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            if l == self.num_layers - 2:
                if use_batch_forward and isinstance(self.sdf_linear, MyLinear):
                    # print('Use Batch Forward')
                    sdf_output = self.sdf_linear.batch_forward(x)
                else:
                    sdf_output = self.sdf_linear(x)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return torch.cat([sdf_output[:, :], x[:, :]], dim=-1)

    def gradient(self, x, use_batch_forward=False):
        x.requires_grad_(True)
        y = self.forward(x, use_batch_forward)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True)[0]
        return gradients

    def dnormal_dw(self, x):
        normal = self.gradient(x, use_batch_forward=True)
        dnormal_dw_list = []
        for i in range(normal.shape[1]):
            normal_now = normal[:, i:i+1]
            d_output = torch.ones_like(normal_now, requires_grad=False, device=normal.device)
            dnormal_dw = torch.autograd.grad(
                outputs=normal_now,
                inputs=self.sdf_linear.weight_matrix,
                grad_outputs=d_output,
                create_graph=True,
            )[0]
            dnormal_dw_list.append(dnormal_dw)
        dnormal_dws = torch.stack(dnormal_dw_list, dim=1)
        return dnormal_dws, normal
    
    def dsdf_dw(self, x):
        sdf = self.get_sdf_vals(x, use_batch_forward=True)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        dsdf_dw = torch.autograd.grad(
            outputs=sdf,
            inputs=self.sdf_linear.weight_matrix,
            grad_outputs=d_output,
            create_graph=True,
        )[0]
        return dsdf_dw

    def get_sdf_vals(self, x, use_batch_forward=False):
        sdf = self.forward(x, use_batch_forward)[:,:1]
        return sdf

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0,
            per_image_code = False,
            with_dn_dw=False,
    ):
        super().__init__()
        self.with_dn_dw = with_dn_dw
        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.per_image_code = per_image_code
        if self.per_image_code:
            # nerf in the wild parameter
            # parameters
            # maximum 1024 images
            self.embeddings = nn.Parameter(torch.empty(1024, 32))
            std = 1e-4
            self.embeddings.data.uniform_(-std, std)
            dims[0] += 32

        print("rendering network architecture:")
        print(dims)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if self.with_dn_dw:
            self.rgb_linear = MyLinear(dims[-2], d_out).cuda()
            if weight_norm:
                self.rgb_linear = nn.utils.weight_norm(self.rgb_linear)
        else:
            self.rgb_linear = nn.Linear(dims[-2], d_out)


        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors, indices, use_batch_forward=False):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        else:
            raise NotImplementedError

        if self.per_image_code:
            image_code = self.embeddings[indices].expand(rendering_input.shape[0], -1)
            rendering_input = torch.cat([rendering_input, image_code], dim=-1)
            
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            if l ==self.num_layers - 2:
                
                if use_batch_forward:
                    x = self.rgb_linear.batch_forward(x)
                else:
                    x = self.rgb_linear(x)

            else:
                lin = getattr(self, "lin" + str(l))
                x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)
        
        x = self.sigmoid(x)
        return x

    def dcolor_dw(self, points, normals, view_dirs, feature_vectors, indices):
        color = self.forward(points, normals, view_dirs, feature_vectors, indices, use_batch_forward=True)
        # print("Color Shape : ", color.shape)
        color_mean = torch.mean(color, dim=-1)
        # print("ColorMean Shape : ", color_mean.shape)
        d_output = torch.ones_like(color_mean, requires_grad=False, device=color_mean.device)
        dcolor_dw = torch.autograd.grad(
            outputs=color_mean,
            inputs=self.rgb_linear.weight_matrix,
            grad_outputs=d_output,
            create_graph=True,
        )[0]
        return color, dcolor_dw


class MonoSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()
        self.with_dn_dw = conf.get_bool('with_dn_dw', default=False)
        Grid_MLP = conf.get_bool('Grid_MLP', default=False)

        self.dn_dw_sdf = conf.get_bool('dn_dw_sdf', default=False)
        self.dn_dw_c = conf.get_bool('dn_dw_c', default=False)
        self.normal_gather = conf.get_bool('normal_gather', default=False)

        self.Grid_MLP = Grid_MLP
        if Grid_MLP:
            assert False, "should no grid"
        else:
            self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0 if self.white_bkgd else self.scene_bounding_sphere, **conf.get_config('implicit_network'))
        
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        
        self.density = LaplaceDensity(**conf.get_config('density'))
        sampling_method = conf.get_string('sampling_method', default="errorbounded")
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))
        

    def forward(self, input, indices):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        
        # we should use unnormalized ray direction for depth
        ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
        depth_scale = ray_dirs_tmp[0, :, 2:]
        
        batch_size, num_pixels, _ = ray_dirs.shape
        # print('batch ', batch_size)
        # print('num pixels ', num_pixels)

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)


        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        if self.normal_gather:
            sdf_nn_output = self.implicit_network.forward(points_flat)
            sdf, feature_vectors = sdf_nn_output[:, :1], sdf_nn_output[:, 1:]
            gradients = self.implicit_network.gradient(points_flat)
            dn_dw_feature = gradients
        
        elif self.with_dn_dw and not self.dn_dw_c:
            if self.dn_dw_sdf:
                
                sdf_nn_output = self.implicit_network.forward(points_flat)
                sdf, feature_vectors = sdf_nn_output[:, :1], sdf_nn_output[:, 1:]
                gradients = self.implicit_network.gradient(points_flat)
                dn_dw_feature = self.implicit_network.dsdf_dw(points_flat)
                # print('Get Feature in SDF Shape : ', dn_dw_feature.shape)
            else:
                sdf_nn_output = self.implicit_network.forward(points_flat)
                sdf, feature_vectors = sdf_nn_output[:, :1], sdf_nn_output[:, 1:]
                dn_dw_feature, gradients = self.implicit_network.dnormal_dw(points_flat)
                # print('Get Feature in Normal Shape : ', dn_dw_feature.shape)
        else:
            sdf_nn_output = self.implicit_network.forward(points_flat)
            sdf, feature_vectors = sdf_nn_output[:, :1], sdf_nn_output[:, 1:]
            gradients = self.implicit_network.gradient(points_flat)
        # sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)
        
        if self.with_dn_dw and self.dn_dw_c:
            rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
            _, dn_dw_feature = self.rendering_network.dcolor_dw(points_flat, gradients, dirs_flat, feature_vectors, indices)
            # print('Get Feature in branch Color Shape : ', dn_dw_feature.shape)
        else:
            rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)

        rgb = rgb_flat.reshape(-1, N_samples, 3)
        # print('RGB Shape : ', rgb.shape)
        # print('DNDW Shape : ', dn_dw_feature.shape)

        weights = self.volume_rendering(z_vals, sdf)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values
        
        # Accumlate dn/dw for jacobi loss
        if self.with_dn_dw or self.normal_gather:
            dn_dw_feature = dn_dw_feature.reshape(*rgb.shape[:-1], -1)
            # print(f'Dn Dw Feature shape : {dn_dw_feature.shape}')
            dn_dw_feature_map = torch.sum(weights.unsqueeze(-1) * dn_dw_feature, 1)
            # print(f'Dn Dw Feature Sum shape : {dn_dw_feature.shape}')
        else:
            dn_dw_feature_map = None

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb':rgb,
            'rgb_values': rgb_values,
            'depth_values': depth_values,
            'z_vals': z_vals,
            'depth_vals': z_vals * depth_scale,
            'sdf': sdf.reshape(z_vals.shape),
            'weights': weights,
            'dn_dw_values': dn_dw_feature_map,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            # add some neighbour points as unisurf
            neighbour_points = eikonal_points + (torch.rand_like(eikonal_points) - 0.5) * 0.01   
            eikonal_points = torch.cat([eikonal_points, neighbour_points], 0)
                   
            grad_theta = self.implicit_network.gradient(eikonal_points)
            
            # split gradient to eikonal points and heighbour ponits
            output['grad_theta'] = grad_theta[:grad_theta.shape[0]//2]
            output['grad_theta_nei'] = grad_theta[grad_theta.shape[0]//2:]
        
        # compute normal map
        normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
        normals = normals.reshape(-1, N_samples, 3)
        normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
        
        # transform to local coordinate system
        rot = pose[0, :3, :3].permute(1, 0).contiguous()
        normal_map = rot @ normal_map.permute(1, 0)
        normal_map = normal_map.permute(1, 0).contiguous()
        
        output['normal_map'] = normal_map

        return output

    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights
