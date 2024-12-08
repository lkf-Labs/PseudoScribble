import torch
import math
from isegm.utils.log import logger


def get_optimizer(model, opt_name, opt_kwargs, fp_lr, fn_lr, seg_lr, other_lr):
    params = []
    if other_lr == 0.0:
        opt_kwargs['lr'] = 0.0
    base_lr = opt_kwargs['lr']
    for name, param in model.named_parameters():
        param_group = {'params': [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue


        if not math.isclose(getattr(param, 'lr_mult', 1.0), 1.0):
            logger.info(f'Applied lr_mult={param.lr_mult} to "{name}" parameter.')
            param_group['lr'] = param_group.get('lr', base_lr) * param.lr_mult

        if 'fp' in name:
            param_group['lr'] = fp_lr * 0.1
        elif 'fn' in name:
            param_group['lr'] = fn_lr * 0.1
        elif 'seg' in name:
            param_group['lr'] = seg_lr * 0.1
        else:
            param_group['lr'] = other_lr * 0.1



        params.append(param_group)

    optimizer = {
        'sgd': torch.optim.SGD,
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW
    }[opt_name.lower()](params, **opt_kwargs)

    return optimizer
