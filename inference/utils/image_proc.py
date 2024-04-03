import os
import numpy as np
from PIL import Image
from crf import CRF
from scipy import ndimage

n_classes = 6
def image_output(mask, realw, realh, key, channel_bindings, output_dir, segmentation_images):
    ######################## COLOR GT #################################
    decoded = decode_segmap(np.array(mask, dtype=np.uint8))
    decoded = Image.fromarray(np.uint8(decoded*255), 'RGBA')
    basewidth = int(realw)
    hsize = int(realh)
    decoded = decoded.resize((basewidth, hsize), Image.LANCZOS)
    decoded.save(os.path.join(output_dir, "{0}_Color_output.png".format(key)))

    if segmentation_images:
        ######################## Primary root GT ###########################
        decoded1 = CRF.decode_channel(mask, [channel_bindings['segmentation']['Primary'],channel_bindings['heatmap']['Seed']])
        decoded1 = Image.fromarray(decoded1).resize((basewidth, hsize), Image.NEAREST).convert('L')
        decoded1.save(os.path.join(output_dir, "{0}_C1.png".format(key)))
        ######################## Lat root GT ###########################
        decoded2 = CRF.decode_channel(mask, channel_bindings['segmentation']['Lateral'])
        decoded2 = Image.fromarray(decoded2).resize((basewidth, hsize), Image.NEAREST).convert('L')
        decoded2.save(os.path.join(output_dir, "{0}_C2.png".format(key)))

def distance_map(mask):
    d = ndimage.distance_transform_edt(mask)
 
    mx = 3
    gamma = 0.1
    epsilon = 0.01
    background_penalty = 10

    d = np.clip(d, 0, mx)
    d = (d - 1) / (mx - 1)
    d = (1 - d) * (gamma - epsilon) + epsilon

    d[(mask == 0)] = background_penalty

    return d
    
def distance_to_weights(mask):
    d = ndimage.distance_transform_edt(mask)

    mx = 2
    gamma = 0.1
    epsilon = 0.01
    background_penalty = 10

    d = np.clip(d, 0, mx)
    d = (d - 1) / (mx - 1)
    d = (1 - d) * (gamma - epsilon) + epsilon

    d[(mask == 0)] = background_penalty

    return d

def decode_segmap(temp):
    Sky = [255, 255, 255,0]
    Building = [0, 114, 178,255]
    Pole = [204, 121, 115,255]
    seed = [213, 94, 0,255]
    tipp = [0, 158, 115,255]
    tipsec = [0, 0, 0, 255] #make it 0,0,0

    label_colours = np.array(
        [
            Sky,
            Building,
            Pole,
            seed,
            tipp,
            tipsec
        ]
    )

    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    a = temp.copy()

    for l in range(0, n_classes):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]
        a[temp == l] = label_colours[l, 3]

    rgba = np.zeros((temp.shape[0], temp.shape[1], 4))
    rgba[:, :, 0] = r / 255.0
    rgba[:, :, 1] = g / 255.0
    rgba[:, :, 2] = b / 255.0
    rgba[:, :, 3] = a / 255.0
    return rgba
