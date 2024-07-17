import torch.nn as nn
import numpy as np

import utils
from .embedder import *
from .density import LaplaceDensity
from .ray_sampler import ErrorBoundSampler

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
            feature_vector_size, # 256
            sdf_bounding_sphere,
            d_in,
            d_out, # SDF
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            embed_type=None,
            sphere_scale=1.0,
            output_activation=None,
            with_dn_dw=False,
            **kwargs
    ):
        super().__init__()

        self.sdf_bounding_sphere = sdf_bounding_sphere
        self.sphere_scale = sphere_scale
        dims = [d_in] + dims + [feature_vector_size]

        self.with_dn_dw = with_dn_dw

        self.embed_fn = None
        if embed_type:
            embed_fn, input_ch = get_embedder(embed_type, input_dims=d_in, **kwargs)
            self.embed_fn = embed_fn
            dims[0] = input_ch
        
        print(f"[INFO] Implicit network dims: {dims}")

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
                if out_dim < 0:
                    print(dims)
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif (embed_type or self.use_grid) and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif (embed_type or self.use_grid) and l in self.skip_in:
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

        # self.sdf_linear = nn.Linear(dims[-2], d_out)
        
        torch.nn.init.normal_(self.sdf_linear.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
        torch.nn.init.constant_(self.sdf_linear.bias, -bias)

        if weight_norm:
            self.sdf_linear = nn.utils.weight_norm(self.sdf_linear)

        self.activation = nn.Softplus(beta=100)
        self.output_activation = None
        if output_activation is not None:
            self.output_activation = activations[output_activation]

    def get_param_groups(self, lr):
        return [{'params': self.parameters(), 'lr': lr}]

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
                x = self.activation(x)

        if self.output_activation is not None:
            x = self.output_activation(x)
        
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
    
    def feature(self, x):
        return self.forward(x)[:,1:]

    def get_outputs(self, x, returns_grad=True, is_training=True):
        x.requires_grad_(returns_grad)
        output = self.forward(x, use_batch_forward=False)
        sdf = output[:,:1]
        feature_vectors = output[:, 1:]
        if returns_grad:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            if self.with_dn_dw and is_training:
                # print('Normal Shape : ', gradients.shape)
                dnormal_dw_list = []
                for i in range(gradients.shape[1]):
                    normal_now = gradients[:, i:i+1]
                    d_output_again = torch.ones_like(normal_now, requires_grad=False, device=gradients.device)
                    dnormal_dw = torch.autograd.grad(
                        outputs=normal_now,
                        inputs=self.sdf_linear.weight_matrix,
                        grad_outputs=d_output_again,
                        create_graph=True,
                        retain_graph=True,
                    )[0]
                    dnormal_dw_list.append(dnormal_dw)
                dnormal_dws = torch.stack(dnormal_dw_list, dim=1)
                # dnormal_dws, normal = self.dnormal_dw(x)
                return sdf, feature_vectors, dnormal_dws, gradients
            else:
                return sdf, feature_vectors, gradients
        else:
            return sdf, feature_vectors, None

    def get_sdf_vals(self, x):
        sdf = self.forward(x)[:,:1]
        ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        return sdf

activations = {
    'sigmoid': nn.Sigmoid(),
    'relu': nn.ReLU(),
    'softplus': nn.Softplus()
}

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            embed_type=None,
            embed_point=None,
            output_activation='sigmoid',
            **kwargs
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]
        self.d_out = d_out

        self.embedview_fn = None
        if embed_type:
            embedview_fn, input_ch = get_embedder(embed_type, input_dims=3, **kwargs)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)
        
        if mode == 'idr':
            self.embedpoint_fn = None
            if embed_point is not None:
                embedpoint_fn, input_ch = get_embedder(input_dims=3, **embed_point)
                self.embedpoint_fn = embedpoint_fn
                dims[0] += (input_ch - 3)

        print(f"[INFO] Rendering network dims: {dims}")
        self.num_layers = len(dims)
        self.weight_norm = weight_norm

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()
        self.output_activation = activations[output_activation]

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        # elif self.mode == 'nerf':
        else:
            rendering_input = torch.cat([view_dirs, feature_vectors], dim=-1)
        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        x = self.output_activation(x)

        return x

