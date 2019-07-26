import sys, os
import torch
import visdom
import argparse
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import cv2
import kdtree

def euclid(pt1, pt2):
    return (pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2

def rrtree(lat, threshold):
    if lat is None or len(lat) == 0:
        return []
        
    tree = kdtree.create(dimensions=2)
    distance_threshold = threshold # 8^2
    for i,pt in enumerate(lat):
        t_pt = (float(pt[0]), float(pt[1]))
        search_result = tree.search_nn(t_pt, dist=euclid)
        if search_result is None:
            tree.add(t_pt)
        else:
            node, dist = search_result[0], search_result[1]
            if dist >= distance_threshold:
                tree.add(t_pt)

    filtered_points = [(int(pt.data[0]), int(pt.data[1])) for pt in kdtree.level_order(tree)]
    return filtered_points