"""
Misc Utility functions
"""
import os
import logging
import datetime
import numpy as np
import torch
import collections

from collections import OrderedDict

def decode_segmap(temp, plot=False):
    Seed = [255, 255, 255]
    P_Root = [0, 255, 0]
    L_Root = [255, 100, 100]
    P_tip = [255, 0, 0]
    L_tip = [147, 0, 227]
    Back = [0, 0, 0]

    label_colours = torch.Tensor(
        [
            Seed,
            P_Root,
            L_Root,
            P_tip,
            L_tip,
            Back,
        ]
    )
    r = temp.clone()
    g = temp.clone()
    b = temp.clone()
    for l in range(0, 6):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = torch.zeros((3, temp.shape[0], temp.shape[1]))
    rgb[0, :, :] = r / 255.0
    rgb[1, :, :] = g / 255.0
    rgb[2, :, :] = b / 255.0
    return rgb

def dict_collate(batch):
    if isinstance(batch[0], collections.Mapping):
        return {key: [d[key] for d in batch] for key in batch[0]}
    elif torch.is_tensor(batch[0]):
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        # This is true when number of threads > 1
        numel = sum([x.numel() for x in batch])
        storage = batch[0].storage()._new_shared(numel)
        out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(batch[0], collections.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return tuple(dict_collate(samples) for samples in transposed)
    else:
        raise TypeError("BAD TYPE", type(batch[0]))

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images 
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value

    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger
