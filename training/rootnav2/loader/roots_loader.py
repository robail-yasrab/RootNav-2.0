from __future__ import absolute_import
import numpy as np
from PIL import Image
import os
import xml.etree.ElementTree
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import cv2
plt.ion()  # interactive mode
from PIL import Image
import torch
import math
import torch.nn as nn
import scipy.misc
import collections
import torchvision
import scipy.misc as m
from torch.utils import data
from .parser import RSMLParser, Plant, Root
from torch.nn.functional import interpolate
from torchvision.transforms.functional import to_tensor
import random
from torchvision.transforms.functional import hflip

sigma=2
label_type='Gaussian'

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

# Gaussian Drawing
cache = {}

def gaussian_kernel_2d(size, sigma):
    gaussian = torch.FloatTensor(size, size)
    centre = (size / 2.0) + 0.5

    twos2 = 2 * math.pow(sigma, 2)

    for x in range(1,size+1):
        for y in range(1,size+1):
            gaussian[y-1,x-1] = -((math.pow(x - centre, 2)) + (math.pow(y - centre, 2))) / twos2

    return gaussian.exp()


def draw_gaussian_2d(img, pt, sigma):
    ptx, pty = int(round(pt[0].item())), int(round(pt[1].item()))
    height, width = img.size(0), img.size(1)

    # Draw a 2D gaussian
    # Check that any part of the gaussian is in-bounds
    tmpSize = math.ceil(3 * sigma)

    ul = [ptx - tmpSize, pty - tmpSize]
    br = [ptx + tmpSize, pty + tmpSize]

    # If not, return the image as is
    if (ul[0] >= width or ul[1] >= height or br[0] < 0 or br[1] < 0):
        return

    # Generate gaussian
    size = 2 * tmpSize + 1
    if size not in cache:
        cache[size] = gaussian_kernel_2d(int(size), sigma)

    g = cache[size]

    # Usable gaussian range
    g_x = [int(max(0, -ul[0])), int(min(size-1, size + (width - 2 - br[0])))]
    g_y = [int(max(0, -ul[1])), int(min(size-1, size + (height - 2 - br[1])))]
    
    # Image range
    img_x = [int(max(0, ul[0])), int(min(br[0], width - 1))]
    img_y = [int(max(0, ul[1])), int(min(br[1], height - 1))]

    sub_img = img[img_y[0]:img_y[1]+1, img_x[0]:img_x[1]+1]
    torch.max(sub_img, g[g_y[0]:g_y[1]+1, g_x[0]:g_x[1]+1], out=sub_img)

def render_heatmap_2d(img, pts, sd):
    if pts is not None:
        for p in pts:
            draw_gaussian_2d(img, p, sd)

# =============================================================================


class rootsLoader(data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        network_input_size = (1024, 1024),
        network_output_size = (512,512),
        hflip=0.0
    ):
        self.root = root
        self.split = split
        self.network_input_size = network_input_size
        self.network_output_size = network_output_size
        self.hflip = hflip
        self.n_classes = 6
        self.files = []

        for filename in os.listdir(root + "/" + split):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')) and '_mask.' not in filename.lower():
                key = os.path.splitext(filename)[0]
                image_path = os.path.join(root, split, filename)
                rsml_path = os.path.join(root, split, key + ".rsml")
                
                if os.path.isfile(rsml_path):
                    # RSML exists for this image, check for existence of cache
                    cache_path = os.path.join(root, split, key + ".pt")
                    if not os.path.isfile(cache_path):
                        print ("Creating cache for", key)
                        self._create_cache_file(image_path, rsml_path, cache_path)

                    self.files.append((image_path, cache_path))

    def _create_cache_file(self, image_path, rsml_path, cache_path):
        source_image = Image.open(image_path)
        source_width, source_height = source_image.size

        # Read RSML file
        # Note: Points are parsed from RSML as x,y not y,x
        plants = RSMLParser.parse(rsml_path, round_points = False)
        
        seeds = []
        primary_tips = []
        lateral_tips = []

        # Round tuple points
        def int_t(t):
            return (int(t[0]), int(t[1]))

        # Create mask image
        mask = np.zeros((source_height,source_width), dtype=np.uint8)

        # Regression points
        for plant in plants:
            seeds.append(plant.seed)
            for r in plant.primary_roots():
                primary_tips.append(r.end)
            
            for r in plant.lateral_roots():
                lateral_tips.append(r.end)

            # DRAW SEED ON MASK
            cv2.circle(mask, int_t((plant.seed)), 10, (5), -1)

        # Segmentation masks
        line_thickness = 4

        # 0 BG
        # 1 Lat seg
        # 2 Lat tip
        # 3 Pri seg
        # 4 Pri tip
        # 5 Seed location

        for plant in plants:
        # Draw lateral roots (ID 2)
            for r in plant.lateral_roots():
                for p in r.pairwise():
                    cv2.line(mask, int_t(p[0]), int_t(p[1]), (1), line_thickness)
                # DRAW CIRCLE FOR LATERAL TIP
                cv2.circle(mask, (int_t(r.end)), 10, (2), -1)

            # Draw primary roots (ID 1)
            for r in plant.primary_roots():
                for p in r.pairwise():
                    cv2.line(mask, int_t(p[0]), int_t(p[1]), (3), line_thickness)
                # DRAW CIRCLE FOR PRI TIP
                cv2.circle(mask, (int_t(r.end)), 10, (4), -1)

        annotation = {
            "seeds": torch.Tensor(seeds),
            "primary": torch.Tensor(primary_tips),
            "lateral": torch.Tensor(lateral_tips),
            "mask": torch.from_numpy(mask).unsqueeze(0)
        }

        torch.save(annotation, cache_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path, cache_path = self.files[index]
        
        image = Image.open(image_path)
        cache = torch.load(cache_path)

        if image.mode == 'RGBA':
            # Split the image into individual bands
            r, g, b, a = image.split()
            # Create a new image without the alpha channel
            image = Image.merge('RGB', (r, g, b))

        # Render heatmap
        y_scale = self.network_output_size[0] / image.height
        x_scale = self.network_output_size[1] / image.width
        scale = torch.Tensor([x_scale, y_scale]) # x,y order

        hm = torch.zeros(3, self.network_output_size[0], self.network_output_size[1]) 
        render_heatmap_2d(hm[0], cache["seeds"].mul(scale), sigma)
        if (cache["primary"].size(0) > 0):
            render_heatmap_2d(hm[1], cache["primary"].mul(scale), sigma)
        if (cache["lateral"].size(0) > 0):
            render_heatmap_2d(hm[2], cache["lateral"].mul(scale), sigma)



        # Resize image and mask to input size
        image = image.resize((self.network_input_size[0], self.network_input_size[1]))

        # IMAGES IS CORRECT SIZE (1024)
        # HM is correct size (512)
        # Mask is native size

        mask = cache["mask"]
        mask = interpolate(mask.unsqueeze(0).float(), (self.network_input_size[0], self.network_input_size[1]), mode='nearest')[0]
        
        # Augmentation - hflip
        if self.hflip > 0.0 and random.random() < self.hflip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.flip(2)
            hm = hm.flip(2)

        # Final resize of mask
        mask = interpolate(mask.unsqueeze(0), (self.network_output_size[0], self.network_output_size[1]), mode='nearest')[0]
        mask = mask.squeeze(0).long()

        # Convert input PIL image to tensor
        image = to_tensor(image)

        # Debugging
        #from torchvision.utils import save_image
        #save_image(image, "source.png")
        #save_image(hm[0], 'seed_hm.png')
        #save_image(hm[1], 'pri_hm.png')
        #save_image(hm[2], 'lat_hm.png')
        #save_image(mask, 'mask.png')

        if self.split == 'test':
            annotations = {
                "seeds": cache["seeds"].mul(scale) if cache["seeds"].size(0) > 0 else torch.Tensor((0)),
                "primary": cache["primary"].mul(scale) if cache["primary"].size(0) > 0 else torch.Tensor((0)),
                "lateral": cache["lateral"].mul(scale) if cache["lateral"].size(0) > 0 else torch.Tensor((0))
            }
            return image, mask, annotations
        else:
            return image, mask, hm
