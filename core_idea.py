import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# A linear class which support batch gradients of parameters
class MyLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input: Tensor) -> Tensor:
        return super().forward(input)

    def batch_forward(self, input: Tensor) -> Tensor:
        bs = input.shape[0]
        self.weight_matrix = self.weight[None, :, :].expand(bs, self.out_features, self.in_features)
        output = torch.add(torch.einsum('ab,abc->ac', input, self.weight_matrix.transpose(1, 2)), self.bias)
        return output

class TinySDFMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TinySDFMLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 256)
        self.sdf_linear = MyLinear(256, out_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sdf_linear.batch_forward(x)
        return x

    def sdf(self, x):
        return self.forward(x)
    
    def normal(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        normals = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True)[0]
        return normals
        
    def dnormal_dw(self, x):
        normal = self.normal(x)
        dnormal_dw_list = []
        for i in range(normal.shape[1]):
            normal_slice = normal[:, i:i+1]
            d_output_slice = torch.ones_like(normal_slice, requires_grad=False, device=normal.device)
            dnormal_dw_slice = torch.autograd.grad(
                outputs=normal_slice,
                inputs=self.sdf_linear.weight_matrix,
                grad_outputs=d_output_slice,
                create_graph=True,
            )[0]
            dnormal_dw_list.append(dnormal_dw_slice)
        return torch.stack(dnormal_dw_list, dim=1)

def mi_loss(pos_condition, feature):
    
    # cosine similarity of feature
    feature = feature.reshape(feature.shape[0], -1)
    feature_dot = (feature @ torch.transpose(feature, 0, 1))
    feature_norm = torch.linalg.norm(feature, dim=-1)
    feature_norm_square = feature_norm * (feature_norm.unsqueeze(1)) + 1e-7
    mi_square = torch.exp(torch.abs(feature_dot / feature_norm_square))
    
    # Sum Results, both positive and negative
    pos_sim_total = (pos_condition.float() * mi_square).sum(dim=-1)
    neg_sim_total = ((~pos_condition).float() * mi_square).sum(dim=-1)
    contrastive = torch.log(pos_sim_total/(pos_sim_total + neg_sim_total) + 1e-7)

    # Loss Calculate
    contrastive_no_nan_count = torch.sum(~torch.isnan(contrastive))
    mi_contrastive_loss = -torch.nansum(contrastive) / contrastive_no_nan_count

    return mi_contrastive_loss

if __name__ == "__main__":
    threshold = 0.9
    
    SDFNet = TinySDFMLP(3, 1)
    
    pts = torch.randn(512, 256, 3) # 512 rays, 256 points per ray, 3D points
    feats = torch.randn(512, 256) # 512 pixels, 256 dim feature
    dots = feats @ feats.t() # compute similarity matrix
    
    norms = torch.linalg.norm(feats, axis=1)
    similarity = dots / (norms[:, None] * norms[None, :])
    positive = similarity > threshold # postive pairs
    print(f'Positive Shape : ', positive.shape)
    
    pts_flattern = pts.view(-1, 3)
    dnormal_dw = SDFNet.dnormal_dw(pts_flattern).reshape(pts.shape[0], pts.shape[1], -1)
    print(f'dnormal_dw Shape : ', dnormal_dw.shape)
    
    feats_bar = torch.sum(dnormal_dw, dim=1)
    # This feats are aggregated by weights function. We use sum for demo
    print(f'Pred Feats Shape : ', feats_bar.shape)
    
    print(f'Mi loss term : {mi_loss(positive, feats)}')
    
    pass