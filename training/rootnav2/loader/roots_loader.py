from __future__ import absolute_import
import pandas as pd
import numpy as np
from PIL import Image
import os
import xml.etree.ElementTree
from PIL import Image
from glob import glob
from skimage import io, transform
from skimage import io, transform
import matplotlib.pyplot as plt
import cv2
plt.ion()  # interactive mode
import pandas as pd
import pandas
from PIL import Image
from PIL import Image
import torch
import torch.nn as nn
import scipy.misc
import collections
import torchvision
import scipy.misc as m
from torch.utils import data
from rootnav2.augmentations import *

sigma=2
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

def gaussian(shape=(3,3),sigma=1):
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
    img = to_numpy(img)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
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
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)

def draw_labelmaps(img, pts, sigma, type='Gaussian'):
    if pts is not None:
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

def color_heatmap(x):
    x = to_numpy(x)
    color = np.zeros((x.shape[0],x.shape[1],3))
    color[:,:,0] = gauss(x, .5, .6, .2) + gauss(x, 1, .8, .3)
    color[:,:,1] = gauss(x, 1, .5, .3)
    color[:,:,2] = gauss(x, 1, .2, .3)
    color[color > 1] = 1
    color = (color * 255).astype(np.uint8)
    return color

def imshow1(img, Name):
    plt.imshow(img)
    plt.axis('off')
    plt.savefig(os.path.join('/home/..../', Name+'.png'), bbox_inches='tight', pad_inches=0)


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

class rootsLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=None,
        augmentations=None,
        img_norm=True,
    ):
        self.root = root
        self.split = split
        self.img_size = [1024, 1024]
        self.img_size1 = [512,512]

        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.mean = np.array([104.00699, 116.66877, 122.67892])
        self.n_classes = 6
        self.files = collections.defaultdict(list)
        for split in ["train", "test", "val"]:
            file_list = os.listdir(root + "/" + split)
            self.files[split] = file_list

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = self.root + "/" + self.split + "/" + img_name
        lbl_path = self.root + "/" + self.split + "annot6/" + img_name[:-4]+'.png'
        TRSML = self.root + "/" + self.split + "RSML/" + img_name[:-4]+'.rsml'
        plants = RSMLParser.parse(TRSML, round_points = True)
        plant = plants[0]
        img = Image.open(img_path)
        lbl = Image.open(lbl_path).convert('L')
        line_thickness = 8
        #################################################

        ################# heat-map ########################
        h, w = img.size 
        new_h, new_w =  self.img_size1 
        new_h, new_w = float(new_h), float(new_w)
        h, w = float(h), float(w)
        a = ((new_h / h), (new_w / w))      
        new_h, new_w = int(new_h), int(new_w)
        gt = np.zeros((5, new_h,new_w), dtype=np.uint8)

        #######################################################
        hm = torch.zeros(3, new_h, new_w)       
        ############### seed ###################     
        aa = plant.seed
        aa = np.multiply(aa, a)
        aa = aa.astype(int)
        aa = np.asarray(aa) 
        cv2.circle(gt[4], (plant.seed), 10, (255, 255, 255), -1)
        hm[2] = draw_labelmap(hm[2], aa, sigma, type=label_type)
     
        ################## pri ###########################
        for r in plant.primary_roots():
			for p in r.pairwise():
				aa = r.end
       			aa = np.multiply(aa, a)
        		aa = aa.astype(int)
        		aa = np.asarray(aa)
        		cv2.line(gt[2], p[0], p[1], (255,255,255), line_thickness) 
        		cv2.circle(gt[3], (r.end), 10, (255, 255, 255), -1)
			hm[1] = draw_labelmap(hm[1], aa, sigma, type=label_type)
   
        ######################latral #######################
        ################## latral ###########################
        for r in plant.lateral_roots():
			for p in r.pairwise():
				aa = r.end
       			aa = np.multiply(aa, a)
        		aa = aa.astype(int)
        		aa = np.asarray(aa) 
		        cv2.line(gt[0], p[0], p[1], (255,255,255), line_thickness) 
        		cv2.circle(gt[1], (r.end), 10, (255, 255, 255), -1)
			hm[0] = draw_labelmap(hm[0], aa, sigma, type=label_type)
       
        ######################latral #######################

        img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        lbl = lbl.resize((self.img_size[0], self.img_size[1]))

        if self.augmentations:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl, hm, gt= self.transform(img, lbl, hm, gt)

        return img, lbl, hm, gt

    def transform(self, img, lbl, hm, gt):

        lbl = lbl.resize((self.img_size1[0], self.img_size1[1]))

        img = np.array(img, dtype=np.uint8)
        img = img.astype(np.float64)

        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(np.array(img)).float()
        lbl = torch.from_numpy(np.array(lbl)).long()
        hm = torch.from_numpy(np.array(hm)).float()
        gt = torch.from_numpy(np.array(gt)).float()
        return img, lbl, hm, gt
    def decode_segmap(self, temp, plot=False):
        Sky = [255, 255, 255]
        Building = [0, 255, 0]
        Pole = [255, 100, 100]
        seed = [255, 0, 0]
        tipp = [147, 0, 227]
        tipsec = [0, 0, 0]


        label_colours = np.array(
            [
                Sky,
                Building,
                Pole,
                seed,
                tipp,
                tipsec,
  
  
            ]
        )
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb


if __name__ == "__main__":
    local_path = "/home/local/datasets/roots"
    augmentations = Compose([RandomRotate(30), RandomHorizontallyFlip(), RandomSizedCrop()])

    dst = rootsLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
