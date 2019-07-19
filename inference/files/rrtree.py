import sys, os
import torch
import visdom
import argparse
import numpy as np
import scipy.misc as misc
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from rtree import index
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm
import cv2
def rrtree(lat):
    trees = []
    output_points= np.asarray(lat,  dtype = np.float32)

    for c in range(len(output_points)):
        idx = index.Index(interleaved=False)
        distance_threshold = 36 # 8^2
        for i,pt in enumerate(output_points[c]):
            neighbour = next(idx.nearest((pt[0],pt[0],pt[1],pt[1])),None)
            if neighbour is None:
                idx.insert(i, (pt[0], pt[0], pt[1], pt[1]))
            else:
                n_pt = output_points[c][neighbour]
                if ((pt[0] - n_pt[0])**2 + (pt[1] - n_pt[1])**2) > distance_threshold:
                    idx.insert(i, (pt[0], pt[0], pt[1], pt[1]))
        trees.append(idx)

        trees= np.asarray(trees)
    input_height = 512
    input_width  = 512
        
    counts = []
    channel_points=[]
    for c in range(len(output_points)):
        channel_points = [output_points[c][pt] for pt in trees[c].intersection((-1,512,-1,512))]
    channel_points = np.asarray(channel_points)
    




    return channel_points