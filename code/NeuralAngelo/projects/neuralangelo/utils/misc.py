'''
-----------------------------------------------------------------------------
Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

from functools import partial
import numpy as np
import torch
import torch.nn.functional as torch_F
import imaginaire.trainers.utils
from torch.optim import lr_scheduler

flip_mat = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])


def get_scheduler(cfg_opt, opt):
    """Return the scheduler object.

    Args:
        cfg_opt (obj): Config for the specific optimization module (gen/dis).
        opt (obj): PyTorch optimizer object.

    Returns:
        (obj): Scheduler
    """
    if cfg_opt.sched.type == 'two_steps_with_warmup':
        warm_up_end = cfg_opt.sched.warm_up_end
        two_steps = cfg_opt.sched.two_steps
        gamma = cfg_opt.sched.gamma

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                if x > two_steps[1]:
                    return 1.0 / gamma ** 2
                elif x > two_steps[0]:
                    return 1.0 / gamma
                else:
                    return 1.0

        scheduler = lr_scheduler.LambdaLR(opt, lambda x: sch(x))
    elif cfg_opt.sched.type == 'cos_with_warmup':
        alpha = cfg_opt.sched.alpha
        max_iter = cfg_opt.sched.max_iter
        warm_up_end = cfg_opt.sched.warm_up_end

        def sch(x):
            if x < warm_up_end:
                return x / warm_up_end
            else:
                progress = (x - warm_up_end) / (max_iter - warm_up_end)
                learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha
                return learning_factor

        scheduler = lr_scheduler.LambdaLR(opt, lambda x: sch(x))
    else:
        return imaginaire.trainers.utils.get_scheduler()
    return scheduler

def jacobi_loss(feature, pos_condition, jacobi_type='vector', no_abs=False):
    # print('Jaco Shape : ', pos_condition.shape) # [bs, n, n]
    # print('Feature Shape : ', feature.shape) # [bs, n, l]
    
    # Dino Space
    if jacobi_type == 'vector':
        # Vector MI Calc
        feature_dot = (feature @ torch.transpose(feature, 1, 2)) # [bs, n, n]
        # print('Feature dot shape : ', feature_dot.shape)
        feature_norm = torch.linalg.norm(feature, dim=-1) # [bs, n]
        # print('Feature norm shape : ', feature_norm.shape)
        feature_norm_square = feature_norm[:, :, None] * (feature_norm[:, None, :]) + 1e-7 # [bs, n, n]
        # print('Feature norm Square shape : ', feature_norm_square.shape)
        if no_abs:
            print('No ABS')
            mi_square = torch.exp(feature_dot / feature_norm_square)
        else:
            mi_square = torch.exp(torch.abs(feature_dot / feature_norm_square))
        
    else: raise NotImplementedError 
    
    # Sum Results
    # print("MI Square Shape : ", mi_square.shape)
    pos_sim_total = (pos_condition.float() * mi_square).sum(dim=-1)
    neg_sim_total = ((~pos_condition).float() * mi_square).sum(dim=-1)
    contrastive = torch.log(pos_sim_total/(pos_sim_total + neg_sim_total) + 1e-7)
    
    # Loss Calculate
    # print(feature_norm)
    jacobi_gradient_loss = torch.mean((feature_norm - 1.0) ** 2)
    contrastive_no_nan_count = torch.sum(~torch.isnan(contrastive))
    mi_contrastive_loss = -torch.nansum(contrastive) / contrastive_no_nan_count

    return jacobi_gradient_loss, mi_contrastive_loss


def eikonal_loss(gradients, outside=None):
    gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2  # [B,R,N]
    gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N]
    if outside is not None:
        return (gradient_error * (~outside).float()).mean()
    else:
        return gradient_error.mean()


def curvature_loss(hessian, outside=None):
    laplacian = hessian.sum(dim=-1).abs()  # [B,R,N]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)  # [B,R,N]
    if outside is not None:
        return (laplacian * (~outside).float()).mean()
    else:
        return laplacian.mean()


def get_activation(activ, **kwargs):
    func = dict(
        identity=lambda x: x,
        relu=torch_F.relu,
        relu_=torch_F.relu_,
        abs=torch.abs,
        abs_=torch.abs_,
        sigmoid=torch.sigmoid,
        sigmoid_=torch.sigmoid_,
        exp=torch.exp,
        exp_=torch.exp_,
        softplus=torch_F.softplus,
        silu=torch_F.silu,
        silu_=partial(torch_F.silu, inplace=True),
    )[activ]
    return partial(func, **kwargs)


def to_full_image(image, image_size=None, from_vec=True):
    # if from_vec is True: [B,HW,...,K] --> [B,K,H,W,...]
    # if from_vec is False: [B,H,W,...,K] --> [B,K,H,W,...]
    if from_vec:
        assert image_size is not None
        image = image.unflatten(dim=1, sizes=image_size)
    image = image.moveaxis(-1, 1)
    return image
