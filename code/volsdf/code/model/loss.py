import torch
from torch import nn
import utils.general as utils


class VolSDFLoss(nn.Module):
    def __init__(self, rgb_loss, eikonal_weight, jacobi_contrastive_weight=0.0):
        super().__init__()
        self.eikonal_weight = eikonal_weight
        self.rgb_loss = utils.get_class(rgb_loss)(reduction='mean')
        self.jacobi_contrastive_weight = jacobi_contrastive_weight

    def get_rgb_loss(self,rgb_values, rgb_gt):
        rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.rgb_loss(rgb_values, rgb_gt)
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_jacobi_loss(self, pos_condition, feature):
        # print('Jaco Shape : ', pos_condition.shape)
        # print('Feature Shape : ', feature.shape)
        
        # Vector MI Calc
        feature = feature.reshape(feature.shape[0], -1)
        feature_dot = (feature @ torch.transpose(feature, 0, 1))
        feature_norm = torch.linalg.norm(feature, dim=-1)
        feature_norm_square = feature_norm * (feature_norm.unsqueeze(1)) + 1e-7
        mi_square = torch.exp(torch.abs(feature_dot / feature_norm_square))
            
        # Sum Results
        pos_sim_total = (pos_condition.float() * mi_square).sum(dim=-1)
        neg_sim_total = ((~pos_condition).float() * mi_square).sum(dim=-1)
        contrastive = torch.log(pos_sim_total/(pos_sim_total + neg_sim_total) + 1e-7)
        
        # Loss Calculate
        jacobi_gradient_loss = torch.mean((feature_norm - 1.0) ** 2)
        contrastive_no_nan_count = torch.sum(~torch.isnan(contrastive))
        mi_contrastive_loss = -torch.nansum(contrastive) / contrastive_no_nan_count
        # return torch.tensor(0.0).float(), torch.tensor(0.0).float()
        return jacobi_gradient_loss, mi_contrastive_loss


    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt)
        if 'grad_theta' in model_outputs:
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        else:
            eikonal_loss = torch.tensor(0.0).cuda().float()

        if 'jaco_info' in ground_truth:
            jaco_gra_loss, jaco_con_loss = self.get_jacobi_loss(ground_truth['jaco_info'], model_outputs['dn_dw_values'])
            # print(f'Jaco Gra loss : {jaco_gra_loss.item()}, Jaco Con Loss : {jaco_con_loss.item()}')
        else:
            jaco_con_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()
            jaco_gra_loss = torch.tensor(0.0, device=model_outputs['rgb_values'].device).float()


        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss +\
               self.jacobi_contrastive_weight * jaco_con_loss

        output = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'jaco_con_loss': jaco_con_loss,
        }

        return output
