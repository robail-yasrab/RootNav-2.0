
import numpy as np
import mahotas
import cv2
import scipy.misc as misc
from PIL import Image

n_classes = 6
def enlarge(mask, realw, realh, name):

    ######################## COLOR GT #################################
    decoded = decode_segmap(np.array(mask, dtype=np.uint8))
    decoded = Image.fromarray(np.uint8(decoded*255))
    basewidth = int(realw)
    wpercent = (basewidth / float(decoded.size[0]))
    hsize = int(realh)
    decoded = decoded.resize((basewidth, hsize), Image.ANTIALIAS)
    decoded.save('./Petri_dish/Results/'+name[:-4]+'_Color_output.png')
    ######################## primery root GT ###########################
    decoded1 = decode_segmap3(np.array(mask, dtype=np.uint8))
    decoded1 = Image.fromarray(np.uint8(decoded1*255))
    decoded1 = decoded1.resize((basewidth, hsize), Image.ANTIALIAS)
    decoded1= decoded1.convert('L') 
    decoded1.save('./Petri_dish/Results/'+name[:-4]+'_C1.png')
    ######################## Lat root GT ###########################
    decoded2 = decode_segmap4(np.array(mask, dtype=np.uint8))
    decoded2 = Image.fromarray(np.uint8(decoded2*255))
    decoded2 = decoded2.resize((basewidth, hsize), Image.ANTIALIAS)  
    decoded2= decoded2.convert('L') 
    decoded2.save('./Petri_dish/Results/'+name[:-4]+'_C2.png') 
def distance_map(decoded):


    decoded= decoded.astype('float32')
    bw = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    bw = bw.astype(np.uint8)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    d = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    #d = np.array([0,1,2,3,4,5,6,7,8,9,10])
    # WHere bw = background
       # Set d from 0.01 to 0.05
    mx = 2
    gamma = 0.5
    epsilon = 0.01
    background_penalty = 10

    d = np.clip(d, 0, mx)
    d = (d - 1) / (mx - 1)
    d = (1 - d) * (gamma - epsilon) + epsilon

    d[(bw == 0)] = background_penalty

    return d

def distance_to_weights(decoded):
    decoded= decoded.astype('float32')
    bw = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
    bw = bw.astype(np.uint8)
    _, bw = cv2.threshold(bw, 40, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    d = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    #d = np.array([0,1,2,3,4,5,6,7,8,9,10])
    # WHere bw = background
       # Set d from 0.01 to 0.05
    mx = 2
    gamma = 0.5
    epsilon = 0.01
    background_penalty = 10

    d = np.clip(d, 0, mx)
    d = (d - 1) / (mx - 1)
    d = (1 - d) * (gamma - epsilon) + epsilon

    d[(bw == 0)] = background_penalty

    return d

def ext_color(decoded_crf, my_list1, my_list2):
    decoded_crf= decoded_crf* 255.0  
    img = np.array(decoded_crf, dtype=np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    maskblue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_green = np.array([50,50,50])
    upper_green = np.array([70,255,255])
    maskgreen = cv2.inRange(hsv, lower_green, upper_green)

    ###########################################################
    #maskgreen[np.where((maskgreen == [255]))] = [128]     # for pri root
    for p,q in zip(my_list1, my_list2):
        p = np.array(p)
        q = np.array(q)
        if p!=[] and q!=[]:
            for x, y in zip(p, q):
                maskgreen[y,x] = 128
            
    #####################################################

    latl =  maskgreen+maskblue

    img2 = np.zeros_like(img)
    img2[:,:,0] = latl
    img2[:,:,1] = latl
    img2[:,:,2] = latl
    #misc.imsave('lat.png', img2)
    return img2


def ext_white_mask(decoded_crf):

    img1 = np.array(decoded_crf, dtype=np.uint8)
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 0], dtype=np.uint8)
    upper_white = np.array([0, 0, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    white = cv2.bitwise_not(img1, img1, mask=white_mask)
    white = np.transpose(white, (2, 0, 1))

    return white

def decode_segmap(temp):
    Back = [255, 255, 255]
    Pri = [213, 94, 0]  
    P_tip = [0,158,115] 
    Lat = [0, 114, 178] 
    L_tipp = [204,121,167] 
    Seed = [0, 0, 0] #make it 0,0,0


    label_colours = np.array(
        [
            Back,
            Pri,
            P_tip,
            Lat,
            L_tipp,
            Seed,


        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
def decode_segmap1(temp):
    back = [0, 0, 0]
    seed = [255, 255, 255]
    priroot = [255, 255, 255]
    seed = [255, 255, 255]
    tipp = [255, 255, 255]
    tipsec = [255, 255, 255] #make it 0,0,0


    label_colours = np.array(
        [
            back,
            seed,
            priroot,
            seed,
            tipp,
            tipsec,


        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def decode_segmap2(temp):
    Sky = [255, 255, 255]
    Building = [0, 255, 0]
    Pole = [0, 255, 0]
    seed = [255, 0, 0]
    tipp = [255, 0, 0]
    tipsec = [0, 255, 0] #make it 0,0,0


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
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb



def decode_segmap3(temp):
    back = [0, 0, 0]
    priroot = [255, 255, 255]
    priroottip = [0, 0, 0]
    lat = [0, 0, 0]
    lattipp = [0, 0, 0]
    seed = [0, 0, 0] #make it 0,0,0


    label_colours = np.array(
        [
            back,
            priroot,
            priroottip,
            lat,
            lattipp,
            seed,


        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb

def decode_segmap4(temp):
    back = [0, 0, 0]
    priroot = [0, 0, 0]
    priroottip = [0, 0, 0]
    lat = [255, 255, 255]
    lattipp = [0, 0, 0]
    seed = [255, 255, 255] #make it 0,0,0


    label_colours = np.array(
        [
            back,
            priroot,
            priroottip,
            lat,
            lattipp,
            seed,


        ]
    )
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb