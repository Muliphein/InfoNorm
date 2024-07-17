import torch.nn as nn
import numpy as np

from utils import rend_util
from model.embedder import *
from model.density import LaplaceDensity
from model.ray_sampler import ErrorBoundSampler
import torch.nn.functional as F

from torch import Tensor
class MyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.in_feature = in_features
        self.out_features = out_features

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    def batch_forward(self, input: Tensor) -> Tensor:
        # print('Batch Forward')
        input_shape = input.shape
        input = input.reshape(-1, input_shape[-1])
        bs = input.shape[0]
        # print('Input Shape : ', input.shape)
        self.weight_matrix = self.weight[None, :, :].expand(bs, self.out_features, self.in_features)
        # print('Matrix Shape : ', self.weight_matrix.shape)
        output = torch.add(torch.einsum('ab,abc->ac', input, self.weight_matrix.transpose(1, 2)), self.bias)
        output = output.reshape(*input_shape[:-1], self.out_features)
        # print('Output Shape : ', self.weight_matrix.shape)
        return output

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size, # 256 for feature
            sdf_bounding_sphere,
            d_in,
            d_out, # 1 for SDF
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
            sphere_scale=1.0,
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
    
    # def get_outputs(self, x):
    #     x.requires_grad_(True)
    #     output = self.forward(x)
    #     sdf = output[:,:1]
    #     ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
    #     if self.sdf_bounding_sphere > 0.0:
    #         sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
    #         sdf = torch.minimum(sdf, sphere_sdf)
    #     feature_vectors = output[:, 1:]
    #     d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
    #     gradients = torch.autograd.grad(
    #         outputs=sdf,
    #         inputs=x,
    #         grad_outputs=d_output,
    #         create_graph=True,
    #         retain_graph=True,
    #         only_inputs=True)[0]

    #     return sdf, feature_vectors, gradients

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
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
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'nerf':
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.sigmoid(x)
        return x

class VolSDFNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.scene_bounding_sphere = conf.get_float('scene_bounding_sphere', default=1.0)
        self.white_bkgd = conf.get_bool('white_bkgd', default=False)
        self.with_dn_dw = conf.get_bool('with_dn_dw', default=False)
        # print('VolSDF with dn_dw : ', self.with_dn_dw)
        self.bg_color = torch.tensor(conf.get_list("bg_color", default=[1.0, 1.0, 1.0])).float().cuda()

        self.implicit_network = ImplicitNetwork(self.feature_vector_size, 0.0, **conf.get_config('implicit_network'))
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))

        self.density = LaplaceDensity(**conf.get_config('density'))
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **conf.get_config('ray_sampler'))

    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        if self.with_dn_dw and self.training:
            sdf_nn_output = self.implicit_network.forward(points_flat)
            sdf, feature_vectors = sdf_nn_output[:, :1], sdf_nn_output[:, 1:]
            dn_dw_feature, gradients = self.implicit_network.dnormal_dw(points_flat)
        else:
            sdf_nn_output = self.implicit_network.forward(points_flat)
            sdf, feature_vectors = sdf_nn_output[:, :1], sdf_nn_output[:, 1:]
            gradients = self.implicit_network.gradient(points_flat)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights = self.volume_rendering(z_vals, sdf)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        output = {
            'rgb_values': rgb_values,
        }

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)

            grad_theta = self.implicit_network.gradient(eikonal_points)
            output['grad_theta'] = grad_theta
            
            # Accumlate dn/dw for jacobi loss
            if self.with_dn_dw:
                # print('Before Dn/Dw feature : ', dn_dw_feature.shape)
                # print('Weight Shape : ', weights.shape)
                dn_dw_feature = dn_dw_feature.reshape(*weights.shape, -1)
                # print(f'Dn Dw Feature shape : {dn_dw_feature.shape}')
                dn_dw_feature_map = torch.sum(weights.unsqueeze(-1) * dn_dw_feature, 1)
                output['dn_dw_values'] = dn_dw_feature_map
                # print('Dn/Dw feature : ', dn_dw_feature_map.shape)

        if not self.training:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)

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
