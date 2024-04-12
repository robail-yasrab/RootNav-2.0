import copy
import functools

from rootnav2.loss.loss import cross_entropy2d

key2loss = {'cross_entropy': cross_entropy2d,
            }

def get_loss_function(cfg):
    if cfg['training']['loss'] is None:
        return cross_entropy2d

    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']
        loss_params = {k:v for k,v in loss_dict.items() if k != 'name'}

        if loss_name not in key2loss:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        return functools.partial(key2loss[loss_name], **loss_params)
