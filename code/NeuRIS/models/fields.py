import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder, positional_encoding_c2f

from torch import Tensor
class MyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, batch_size: int = 0, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    def batch_forward(self, input: Tensor) -> Tensor:
        bs = input.shape[0]
        self.weight_matrix = self.weight[None, :, :].expand(bs, self.out_features, self.in_features)
        output = torch.add(torch.einsum('ab,abc->ac', input, self.weight_matrix.transpose(1, 2)), self.bias)
        return output


class SDFNetwork(nn.Module):
    def __init__(self,
                 d_in,
                 d_out, # D Feature
                 d_sdf, # D SDF
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 activation='softplus',
                 reverse_geoinit = False,
                 use_emb_c2f = False,
                 emb_c2f_start = 0.1,
                 emb_c2f_end = 0.5,
                 dn_dp_type='sum',
                 batch_size=512):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        self.dn_dp_type = dn_dp_type

        self.multires = multires
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in, normalize=False)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch
        logging.info(f'SDF input dimension: {dims[0]}')

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.use_emb_c2f = use_emb_c2f
        if self.use_emb_c2f:
            self.emb_c2f_start = emb_c2f_start
            self.emb_c2f_end = emb_c2f_end
            logging.info(f"Use coarse-to-fine embedding (Level: {self.multires}): [{self.emb_c2f_start}, {self.emb_c2f_end}]")

        self.alpha_ratio = 0.0

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if reverse_geoinit:
                        logging.info(f"Geometry init: Indoor scene (reverse geometric init).")
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                    else:
                        logging.info(f"Geometry init: DTU scene (not reverse geometric init).")
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
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

        # sdf_output
        self.sdf_linear = MyLinear(dims[-2], d_sdf, batch_size=batch_size)
        if reverse_geoinit:
            logging.info(f"Geometry init: Indoor scene (reverse geometric init).")
            torch.nn.init.normal_(self.sdf_linear.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[-2]), std=0.0001)
            torch.nn.init.constant_(self.sdf_linear.bias, bias)
        else:
            logging.info(f"Geometry init: DTU scene (not reverse geometric init).")
            torch.nn.init.normal_(self.sdf_linear.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[-2]), std=0.0001)
            torch.nn.init.constant_(self.sdf_linear.bias, -bias)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

        self.weigth_emb_c2f = None
        self.iter_step = 0
        self.end_iter = 3e5

    def forward(self, inputs, use_batch_forward=False):
        inputs = inputs * self.scale

        if self.use_emb_c2f and self.multires > 0:
            inputs, weigth_emb_c2f = positional_encoding_c2f(inputs, self.multires, emb_c2f=[self.emb_c2f_start, self.emb_c2f_end], alpha_ratio = (self.iter_step / self.end_iter))
            self.weigth_emb_c2f = weigth_emb_c2f
        elif self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)
        else:
            NotImplementedError

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            if l == self.num_layers - 2:
                
                if use_batch_forward and isinstance(self.sdf_linear, MyLinear):
                    # print('Use Batch Forward')
                    sdf_output = self.sdf_linear.batch_forward(x)
                else:
                    sdf_output = self.sdf_linear(x)

            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)

        return torch.cat([sdf_output[:, :] / self.scale, x[:, :]], dim=-1)

    def sdf(self, x, use_batch_forward=False):
        return self.forward(x, use_batch_forward)[:, :1]

    def sdf_hidden_appearance(self, x, use_batch_forward=False):
        return self.forward(x, use_batch_forward)

    def gradient(self, x, use_batch_forward=False):
        x.requires_grad_(True)
        y = self.sdf(x, use_batch_forward)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True)[0]
        return gradients

    def dsdf_dw(self, x):
        sdf = self.sdf(x, use_batch_forward=True)
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        dsdf_dw = torch.autograd.grad(
            outputs=sdf,
            inputs=self.sdf_linear.weight_matrix,
            grad_outputs=d_output,
            create_graph=True,
        )[0]
        return dsdf_dw

    def dnormal_dw(self, x):
        normal = self.gradient(x, use_batch_forward=True)
        if self.dn_dp_type == 'sum':
            d_output = torch.ones_like(normal, requires_grad=False, device=normal.device)
            dnormal_dw = torch.autograd.grad(
                outputs=normal,
                inputs=self.sdf_linear.weight_matrix,
                grad_outputs=d_output,
                create_graph=True,
            )[0]
            return dnormal_dw, normal
        elif self.dn_dp_type == 'single':
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
                
            return torch.stack(dnormal_dw_list, dim=1), normal
        else:
            raise NotImplementedError(f"No Such dn/dp Type {self.dn_dp_type}")

class FixVarianceNetwork(nn.Module):
    def __init__(self, base):
        super(FixVarianceNetwork, self).__init__()
        self.base = base
        self.iter_step = 0

    def set_iter_step(self, iter_step):
        self.iter_step = iter_step

    def forward(self, x):
        return torch.ones([len(x), 1]) * np.exp(-self.iter_step / self.base)

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val=1.0, use_fixed_variance = False):
        super(SingleVarianceNetwork, self).__init__()
        if use_fixed_variance:
            logging.info(f'Use fixed variance: {init_val}')
            self.variance = torch.tensor([init_val])
        else:
            self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            d_feature,
            mode,
            d_in,
            d_out,
            d_hidden,
            n_layers,
            weight_norm=True,
            multires_view=0,
            squeeze_out=True,
            feature_level=None,
    ):
        super().__init__()

        self.feature_level = feature_level
        self.mode = mode
        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

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

        self.rgb_linear = MyLinear(dims[-2], d_out)
        if weight_norm:
            self.rgb_linear = nn.utils.weight_norm(self.rgb_linear)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors, use_batch_forward=False):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = None
        feature_output = None

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

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

            if self.feature_level is not None:
                if l == self.num_layers - 2 - self.feature_level:
                    feature_output = x

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x, feature_output

    def dcolor_dw(self, points, normals, view_dirs, feature_vectors):
        color, feature_output = self.forward(points, normals, view_dirs, feature_vectors, use_batch_forward=True)
        color_mean = torch.mean(color, dim=-1)
        d_output = torch.ones_like(color_mean, requires_grad=False, device=color_mean.device)
        dcolor_dw = torch.autograd.grad(
            outputs=color_mean,
            inputs=self.rgb_linear.weight_matrix,
            grad_outputs=d_output,
            create_graph=True,
        )[0]
        return color, dcolor_dw, feature_output

# Code from nerf-pytorch
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, d_in=3, d_in_view=3, multires=0, multires_view=0, output_ch=4, skips=[4], use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.d_in_view = d_in_view
        self.input_ch = 3
        self.input_ch_view = 3
        self.embed_fn = None
        self.embed_fn_view = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in, normalize=False)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        if multires_view > 0:
            embed_fn_view, input_ch_view = get_embedder(multires_view, input_dims=d_in_view, normalize=False)
            self.embed_fn_view = embed_fn_view
            self.input_ch_view = input_ch_view

        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in
                                        range(D - 1)])

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_view + W, W // 2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts, input_views):
        if self.embed_fn is not None:
            input_pts = self.embed_fn(input_pts)
        if self.embed_fn_view is not None:
            input_views = self.embed_fn_view(input_views)

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            return alpha + 1.0, rgb
        else:
            assert False
