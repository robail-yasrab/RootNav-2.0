from rootnav2.schedulers.schedulers import *

key2scheduler = {'constant_lr': ConstantLR,
                 'poly_lr': PolynomialLR,
                 'multi_step': MultiStepLR,
                 'cosine_annealing': CosineAnnealingLR,
                 'exp_lr': ExponentialLR}


def get_scheduler(optimizer, scheduler_dict):
    if scheduler_dict is None:
        return ConstantLR(optimizer)
    
    s_type = scheduler_dict['name']
    scheduler_dict.pop('name')

    warmup_dict = {} 
    if 'warmup_iters' in scheduler_dict:
        # This can be done in a more pythonic way... 
        warmup_dict['warmup_iters'] = scheduler_dict.get('warmup_iters', 100)
        warmup_dict['mode'] = scheduler_dict.get('warmup_mode', 'linear')
        warmup_dict['gamma'] = scheduler_dict.get('warmup_factor', 0.2)

        scheduler_dict.pop('warmup_iters', None) 
        scheduler_dict.pop('warmup_mode', None)
        scheduler_dict.pop('warmup_factor', None) 

        base_scheduler = key2scheduler[s_type](optimizer, **scheduler_dict)
        return WarmUpLR(optimizer, base_scheduler, **warmup_dict)

    return key2scheduler[s_type](optimizer, **scheduler_dict)
