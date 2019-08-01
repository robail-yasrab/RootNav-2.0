import operator
import os
import sys
import yaml
import time
import math
import torch
import shutil
import random
import argparse
import datetime
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import scipy.misc as misc
#from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils import data
from tqdm import tqdm
import cv2

#weights = [0.5794, 3.6466, 0.0041, 25.7299]

#######################################################################
sigma=1
label_type='Gaussian'
class RSMLParser():
    @staticmethod
    def parse(path, round_points = False):
        e = xml.etree.ElementTree.parse(path).getroot()
        metadata = e.find('metadata')
        scene = e.find('scene')
        # Only returns plants in current implementation
        return [Plant(p, round_points) for p in scene.findall('plant')]

class Plant():
    def __init__(self, xml_node, round_points = False):
        assert(xml_node.tag == 'plant')
        self.id = xml_node.attrib.get('ID')
        self.label = xml_node.attrib.get('label')
        self.roots = [Root(child_node, round_points) for child_node in xml_node.findall('root')]

        self.seed = self.roots[0].start if self.roots else None

    def all_roots(self):
        for r in self.roots:
            # Return current primary
            yield r
            # Return all child roots
            for c in r.roots:
                yield c

    def primary_roots(self):
        for r in self.roots:
            # Return only primary
            yield r

    def lateral_roots(self):
        for r in self.roots:
            # Return only child roots
            for c in r.roots:
                yield c

class Root():
    def __init__(self, xml_node, round_points = False):
        assert(xml_node.tag == 'root')
        self.id = xml_node.attrib.get('ID')
        self.label = xml_node.attrib.get('label')
        
        self.points = [(float(p.attrib['x']), float(p.attrib['y'])) for p in xml_node.find('geometry').find('polyline')]
        
        if round_points:
            self.points = [(int(round(p[0])),int(round(p[1]))) for p in self.points]

        self.roots = [Root(child_node, round_points) for child_node in xml_node.findall('root')]

        self.start = self.points[0] if self.points else None
        self.end = self.points[-1] if self.points else None

    def pairwise(self):
        return zip(self.points[:-1],self.points[1:])

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img

def im_to_torch(img):
    img = np.transpose(img, (2, 0, 1)) # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def load_image(img_path):
    # H x W x C => C x H x W
    return scipy.misc.imread(img_path, mode='RGB')

def resize(img, owidth, oheight):
    img = im_to_numpy(img)
    print('%f %f' % (img.min(), img.max()))
    img = scipy.misc.imresize(
            img,
            (oheight, owidth)
        )
    img = im_to_torch(img)
    print('%f %f' % (img.min(), img.max()))
    return img

# =============================================================================
# Helpful functions generating groundtruth labelmap 
# =============================================================================

def gaussian(shape=(1,1),sigma=1):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    return to_torch(h).float()

def draw_labelmap(img, pt, sigma, type='Gaussian'):
    # Draw a 2D gaussian 
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = to_numpy(img)
     # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[2] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    elif type == 'Cauchy':
        g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[2]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[1]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[2])
    img_y = max(0, ul[1]), min(br[1], img.shape[1])

    img[:,img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)

def draw_labelmaps(img, pts, sigma, type='Gaussian'):
    if pts is not None:
        a =  len(pts)
        for p in pts:
            #print (p)
            #break;
            ppp = draw_labelmap(img, p, sigma, type)

    return ppp


# =============================================================================
# Helpful display functions
# =============================================================================

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d


def imshow1(img, Name):
    #npimg = im_to_numpy(img*255).astype(np.uint8)
    #print npimg.shape 
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join('/home/pszry/usr/local/SegNet/Dataset/Heatmap/out/', Name+'.png'), bbox_inches='tight', pad_inches=0)


def show_joints(img, pts):
    imshow(img)
    
    for i in range(pts.size(0)):
        if pts[i, 2] > 0:
            plt.plot(pts[i, 0], pts[i, 1], 'yo')
    plt.axis('off')

def show_sample(inputs, target):
    num_sample = inputs.size(0)
    num_joints = target.size(1)
    height = target.size(2)
    width = target.size(3)

    for n in range(num_sample):
        inp = resize(inputs[n], width, height)
        out = inp
        for p in range(num_joints):
            tgt = inp*0.5 + color_heatmap(target[n,p,:,:])*0.5
            out = torch.cat((out, tgt), 2)
        
        imshow(out)
        plt.show()

def sample_with_heatmap(inp, out, num_rows=2, parts_to_show=None):
    inp = to_numpy(inp * 255)
    out = to_numpy(out)

    img = np.zeros((inp.shape[1], inp.shape[2], inp.shape[0]))
    for i in range(3):
        img[:, :, i] = inp[i, :, :]

    if parts_to_show is None:
        parts_to_show = np.arange(out.shape[0])

    # Generate a single image to display input/output pair
    num_cols = int(np.ceil(float(len(parts_to_show)) / num_rows))
    size = img.shape[0] // num_rows

    full_img = np.zeros((img.shape[0], size * (num_cols + num_rows), 3), np.uint8)
    full_img[:img.shape[0], :img.shape[1]] = img

    inp_small = scipy.misc.imresize(img, [size, size])

    # Set up heatmap display for each part
    for i, part in enumerate(parts_to_show):
        part_idx = part
        out_resized = scipy.misc.imresize(out[part_idx], [size, size])
        out_resized = out_resized.astype(float)/255
        out_img = inp_small.copy() * .3
        color_hm = color_heatmap(out_resized)
        out_img += color_hm * .7

        col_offset = (i % num_cols + num_rows) * size
        row_offset = (i // num_cols) * size
        full_img[row_offset:row_offset + size, col_offset:col_offset + size] = out_img

    return full_img

def batch_with_heatmap(inputs, outputs, mean=torch.Tensor([0.5, 0.5, 0.5]), num_rows=2, parts_to_show=None):
    batch_img = []
    for n in range(min(inputs.size(0), 4)):
        inp = inputs[n] + mean.view(3, 1, 1).expand_as(inputs[n])
        batch_img.append(
            sample_with_heatmap(inp.clamp(0, 1), outputs[n], num_rows=num_rows, parts_to_show=parts_to_show)
        )
    return np.concatenate(batch_img)

######################################################################
def imshow1(img, Name):
    #npimg = im_to_numpy(img*255).astype(np.uint8)
    #print npimg.shape 
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join('/home/pszry/usr/local/SegNet/Dataset/12-11-2018/pytorch-semseg/', Name+'.png'), bbox_inches='tight', pad_inches=0)

def gauss(x, a, b, c, d=0):
    return a * np.exp(-(x - b)**2 / (2 * c**2)) + d

def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x[:,:,0], .5, .6, .2) + gauss(x[:,:,0], 1, .8, .3)
    color[:,:,1] = gauss(x[:,:,1], 1, .5, .3)
    color[:,:,2] = gauss(x[:,:,2], 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color
############################ NMS Prop ################################
def _distance_squared(a, b):
    return pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2)

def get_distances(pt, gt):
    d = torch.zeros(len(pt))

    for i in range((len(pt))):
        mn = 10000
        for j in range(len(gt)):
            dist = _distance_squared(pt[i], gt[j])
            if dist < mn:
                mn = dist
        d[i] = math.sqrt(mn)

    return d
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor
def evaluate_heatmap(heatmap, gtpoints, nmsthreshold, distancethreshold):
    prpoints = nonmaximalsuppression(heatmap, nmsthreshold)
    #print prpoints

    if len(prpoints) == 0 or len(gtpoints) == 0:
        # Empty tensor, either early in the training process, or an empty image
        if len(prpoints) == len(gtpoints):
            # No predicted or target points
            return 0, 0, 0
        elif len(prpoints) == 0:
            # No predicted points, all points are false negatives
            return 0, 0, len(gtpoints)
        else:
            # No grond truth points, all false positives
            return 0, len(prpoints), 0

    prdist = get_distances(prpoints, gtpoints).le_(distancethreshold)
    gtdist = get_distances(gtpoints, prpoints).le_(distancethreshold)

    tp = int(prdist.sum())
    fp = int((1 - prdist).sum())
    fn = int((1 - gtdist).sum())

    return tp, fp, fn, prpoints


    

def evaluate_heatmap_batch(heatmap, gtpoints, nmsthreshold, distancethreshold):
    batch_count = heatmap.size(0)
    channel_count = heatmap.size(1)

    #if distancethresholds.size(1) != channel_count:
    #    raise Exception("Incorrect number of distance thresholds")

    channel_results = []
    channel_results1 = []

    for channel in range(0,channel_count):
        ctp, cfp, cfn = 0, 0, 0
        for batch in range(0,batch_count):
            if len(gtpoints[batch]) != channel_count:
                raise Exception("Number of ground truth channels does not match heatmap channels")

            tp, fp, fn, prpoints = evaluate_heatmap(heatmap[batch][channel], gtpoints[batch][channel], nmsthreshold, distancethreshold[batch][channel])

            ctp += tp
            cfp += fp
            cfn += fn
        channel_results1.append([prpoints])   
        channel_results.append([ctp, cfp, cfn])

    return channel_results, channel_results1


def nonmaximalsuppression(tensor, threshold):
    pred_data = tensor.storage()
    offset = tensor.storage_offset()
    stride = int(tensor.stride()[0])
    numel = tensor.numel()
    points = []

    # Corners
    val = pred_data[0 + offset]
    if val >= threshold and val >= pred_data[1 + offset] and val >= pred_data[stride + offset]:
        points.append([0, 0])

    val = pred_data[stride - 1 + offset]
    if val >= threshold and val >= pred_data[stride - 2 + offset] and val >= pred_data[2 * stride - 1 + offset]:
        points.append([stride - 1, 0])
        
    val = pred_data[numel - stride + offset]
    if val > threshold and val >= pred_data[numel - stride + 1 + offset] and val >= pred_data[numel - 2 * stride + offset]:
        points.append([0, stride - 1])

    val = pred_data[numel - 1 + offset]
    if val > threshold and val >= pred_data[numel -2 + offset] and val >= pred_data[numel - 1 - stride + offset]:
        points.append([stride - 1, stride - 1])

    # Top y==0
    for i in range(1,stride-1):
        i += offset
        val = pred_data[i]
        if val >= threshold and val >= pred_data[i-1] and val >= pred_data[i+1] and val >= pred_data[i+stride]:
            points.append([i - offset, 0])

    # Bottom y==stride-1
    for i in range(numel-stride+1,numel-1):
        i += offset
        val = pred_data[i]
        if val >= threshold and val >= pred_data[i-1] and val >= pred_data[i+1] and val >= pred_data[i-stride]:
            points.append([i - numel + stride - offset, stride])

    # Front x==0
    for i in range(stride, stride * (stride - 1), stride):
        i += offset
        val = pred_data[i]
        if val >= threshold and val >= pred_data[i+stride] and val >= pred_data[i-stride] and val >= pred_data[i+1]:
            points.append([0, (i - offset) // stride])

    # Back x == stride-1
    for i in range(stride - 1, stride * (stride - 1), stride):
        i += offset
        val = pred_data[i]
        if val >= threshold and val >= pred_data[i+stride] and val >= pred_data[i-stride] and val >= pred_data[i-1]:
            points.append([stride - 1, (i - offset) // stride])

    # Remaining inner pixels
    for i in range(stride+1, stride * (stride - 1), stride):
        for j in range(i,i+stride-2):
            j += offset
            val = pred_data[j]
            if val >= threshold and val >= pred_data[j+1] and val >= pred_data[j-1] and val >= pred_data[j+stride] and val >= pred_data[j-stride]:
                points.append([(j - offset) % stride, i // stride])

    return points
##########################Image DRAW #################################

def show_example(img, gt_mask, pred_mask, nn):

    img_np = img.cpu().data.numpy()

    img_np = np.transpose(img_np, [1,2,0])
    #img_np += 128
    #img_np = cv2.resize(img_np, (512,512))
    #print img_np.shape 
    #gt_mask_np = gt_mask.cpu().data.numpy()

    gt_mask_np = np.transpose(gt_mask, [1,2,0]) * 255.0   
    gt_mask_np = np.repeat(gt_mask_np, 3, 2)
    #gt_mask_np = cv2.resize(gt_mask_np, (512,512))

    pred_mask = np.asarray(pred_mask) #.cpu().numpy()
    #pred_mask = np.transpose(pred_mask, [1,2,0]) * 255.0
    #pred_mask = np.repeat(pred_mask, 3, 2)
    #pred_mask = cv2.resize(pred_mask, (512,512))
    
    #print img_np.shape, gt_mask_np.shape, pred_mask.shape, nn.shape
    #img = np.zeros((512,2048,3))
    #img[0:512,0:512,:] = img_np[:,:,:]
    #img[0:512,512:1024,:] = gt_mask_np[:,:,:]
    #img[0:512,1024:1536,:] = pred_mask[:,:,:]
    #img[0:512,1536:2048,:] = nn[:,:,:]

    # img = np.concatenate((img_np, gt_mask_np, pred_mask, nn), axis=1)
    img = [img_np, gt_mask_np, pred_mask, nn]


    #fig = plt.figure()
    fig = plt.figure(figsize=(30, 30))
    im1 = fig.add_subplot(141)
    im2 = fig.add_subplot(142)
    im3 = fig.add_subplot(143)
    im4 = fig.add_subplot(144)
    im1.imshow(img_np)
    im1.axis('off')
    im2.imshow(gt_mask_np)
    im2.axis('off')
    im3.imshow(pred_mask)
    im3.axis('off')
    im4.imshow(nn)
    im4.axis('off')


    plt.xticks([]), plt.yticks([]) 
    plt.savefig('t.jpg', bbox_inches="tight", pad_inches=0)


    ##it also works -- with small image size#
    # for i in range(4):
    #     plt.subplot(1,4,i+1)
    #     plt.imshow(img[i])
    #     plt.axis('off')
    # plt.savefig('t.jpg',  aspect="auto")


    #print img.shape 
    #fig = plt.figure(figsize=(30, 30))
    #plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    #plt.savefig('4pic', bbox_inches="tight", pad_inches=0)
    #cv2.namedWindcdow('i', img.astype(np.uint8))
    #cv2.waitKey(5)
###############################################################
    
from numpy.lib.stride_tricks import as_strided
import cv2

import numpy as np
def neighbors(im, i, j, d=1):
    n = im[i-d:i+d+1, j-d:j+d+1].flatten()
    # remove the element (i,j)
    n = np.hstack((n[:len(n)//2],n[len(n)//2+1:] ))
    return n
def sliding_window(arr, window_size):
    """ Construct a sliding window view of the array"""
    arr = np.asarray(arr)
    window_size = int(window_size)
    if arr.ndim != 2:
        raise ValueError("need 2-D input")
    if not (window_size > 0):
        raise ValueError("need a positive window size")
    shape = (arr.shape[0] - window_size + 1,
             arr.shape[1] - window_size + 1,
             window_size, window_size)
    if shape[0] <= 0:
        shape = (1, shape[1], arr.shape[0], shape[3])
    if shape[1] <= 0:
        shape = (shape[0], 1, shape[2], arr.shape[1])
    strides = (arr.shape[1]*arr.itemsize, arr.itemsize,
               arr.shape[1]*arr.itemsize, arr.itemsize)
    return as_strided(arr, shape=shape, strides=strides)

def cell_neighbors(arr, i, j, d):
    """Return d-th neighbors of cell (i, j)"""
    w = sliding_window(arr, 2*d+1)

    ix = np.clip(i - d, 0, w.shape[0]-1)
    jx = np.clip(j - d, 0, w.shape[1]-1)

    i0 = max(0, i - d - ix)
    j0 = max(0, j - d - jx)
    i1 = w.shape[2] - max(0, d - i + ix)
    j1 = w.shape[3] - max(0, d - j + jx)

    return w[ix, jx][i0:i1,j0:j1].ravel()


