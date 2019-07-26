import cv2
import time
import sys, os
import torch
import argparse
import numpy as np
import scipy.misc as misc
import torch.nn.functional as F
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from files.hourglass import hg
from files.utils import convert_state_dict
from files.func import *
from files.astar import *
from files.rrtree import rrtree
from files.rsml import prettify
from files.image_proc import *
from files.AStar_gaps import *
from files.AStar_gaps_laterals import *
from glob import glob
from rsml import Spline, Plant, Root, RSMLWriter
from models import ModelLoader


n_classes = 6
img_size,img_size1 = 1024, 512
t0 = time.time()
fileExtensions = [ "jpg", "JPG", "png", "tif" ]

def run_rootnav(model_data, use_cuda, input_dir, output_dir):
    
    # Load parameters
    model = model_data['model']
    multi_plant = model_data['multi-plant']
    primary_spline_params = model_data['spline-config']['primary']
    lateral_spline_params = model_data['spline-config']['lateral']

    for extension in fileExtensions:
        files = glob(os.path.join(input_dir, "*." + extension))
        
        for file in files:
            name = os.path.basename(file) 
            key = os.path.splitext(name)[0]

            print 'Now Reading',name
            sys.stdout.flush()
            img = misc.imread(file)

            ####################### RESIZE #########################################
            realh, realw= img.shape[:2]
            realw= float(realw)
            realh= float(realh)
            ########################################################################
            factor1 =realh/512
            factor2 = (realw)/512
            factor1= float(factor1)
            factor2= float(factor2)
            ##########################################################################        
            resized_img = misc.imresize(img, (img_size1, img_size1), interp='bicubic')

            orig_size = img.shape[:-1]

            img = misc.imresize(img, (img_size, img_size))
            img = img.astype(float) / 255.0
            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()

            if use_cuda:
                model.cuda(0)
                images = Variable(img.cuda(0), requires_grad=True)
            else:
                images = Variable(img, requires_grad=True)

            ######################## MODEL FORWARD #################################
            m = model(images)

            res1 = m[-1]
            #m = m.unsqueeze(0)
            out3= res1[:,3:4,:,:] # sec root          
            out1= res1[:,1:2,:,:] # pri root 
            out0= res1[:,0:1,:,:] # back
            
            out5= res1[:,5:6,:,:] # seed         
            out4= res1[:,4:5,:,:] # pri tip
            out2= res1[:,2:3,:,:] # latral tip    
            o = torch.cat((out2, out4, out5), 1)
            o1 = torch.cat((out0, out1, out3), 1)

            output_heatmap = o.data.cpu()
            batch_count = output_heatmap.size(0)
            channel_count = output_heatmap.size(1)
            channel_results = []
            for channel in range(0,channel_count):
                for batch in range(0,batch_count):
                    prpoints = nonmaximalsuppression(output_heatmap[batch][channel], 0.7)
                    channel_results.append([prpoints])  

            a4 = channel_results[2]
            a4 = np.asarray(a4) #seed
            a = a4.squeeze(0)
           
            a5 = channel_results[1]
            a5 = np.asarray(a5) #pri tip
            a1 = a5.squeeze(0)
                        
            a6 = channel_results[0]
            a6 = np.asarray(a6).squeeze(0) # latrl tip

            n = F.softmax(res1, dim=1)

            ########################### CRF #################################
            unary = n.data.cpu().numpy()
            unary = np.squeeze(unary, 0)
            unary = -np.log(unary)
            unary = unary.transpose(2, 1, 0)
            w, h, c = unary.shape
            #print c.shape 
            unary = unary.transpose(2, 0, 1).reshape(6, -1)
            unary = np.ascontiguousarray(unary)
           
            resized_img = np.ascontiguousarray(resized_img)

            d = dcrf.DenseCRF2D(w, h, 6)
            d.setUnaryEnergy(unary)
            d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=resized_img, compat=1)
            q = d.inference(50)
            mask = np.argmax(q, axis=0).reshape(w, h).transpose(1, 0)
            enlarge(mask, realw, realh, key, output_dir)
            decoded_crf = decode_segmap(np.array(mask, dtype=np.uint8))

            pred = np.squeeze(res1.data.max(1)[1].cpu().numpy(), axis=0)
            decoded = decode_segmap1(pred) 
            decoded= np.asarray(decoded, dtype=np.float32)
            ###################################################################

            ######################## COLOR GT #################################

            ######################## PRIMARY ROOT ###########################
            
            # Filter seed and primary tip locations
            seed_locations = rrtree(a4.squeeze(), 36)
            primary_tips = rrtree(a5.squeeze(), 36)

            start = seed_locations[0]
            
            decoded = np.asarray(decoded, dtype = np.float32)
            gray_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            gray_image = Image.fromarray(gray_image)
            path_pixels = gray_image.load()
            distance = manhattan
            heuristic = manhattan
            pri_gt_mask = decode_segmap4(np.array(mask, dtype=np.uint8))
            weights = distance_map(pri_gt_mask) #distance map
            
            lateral_goal_dict = {}

            # Create Plant structure
            plant = Plant(1, 'wheat', seed=start)

            # Search across primary roots
            for tip in primary_tips:
                path = AStar_Pri(start, tip, von_neumann_neighbors, distance, heuristic, weights)
                if path !=[]:
                    scaled_primary_path = [(x*factor2,y*factor1) for (x,y) in path]
                    plant.roots.append(Root(scaled_primary_path, spline_tension = primary_spline_params['tension'], spline_knot_spacing = primary_spline_params['spacing']))
                    current_pid = len(plant.roots) - 1
                    for pt in path:
                        lateral_goal_dict[pt] = current_pid

            lat_gt_mask = decode_segmap3(np.array(mask, dtype=np.uint8))           
            img3= distance_to_weights(lat_gt_mask)            

            # Filter candidate lateral root tips
            lateral_tips = rrtree(a6, 36)

            # Search across lateral roots
            for idxx, i in enumerate(lateral_tips):
                path, pid = AStar_Lat(i, lateral_goal_dict, von_neumann_neighbors, distance, heuristic, img3)
                if path !=[]:
                    scaled_lateral_path = [(x*factor2,y*factor1) for (x,y) in reversed(path)]
                    lateral_root = Root(scaled_lateral_path, spline_tension = lateral_spline_params['tension'], spline_knot_spacing = lateral_spline_params['spacing'])
                    plant.roots[pid].roots.append(lateral_root)

            # Output to RSML
            RSMLWriter.save(key, output_dir, [plant])

            ############################# Total time per Image ######################
            print("Dense CRF Post Processed Mask and RSML File is Saved at: Blue_paper/Results/")
            t1 = time.time()
            total = t1-t0
            print ("RSML Time elapsed:", total, "\n")

def list_models():
    print ("Model     \t\tDescription")
    for name, desc in ModelLoader.list_models(True):
        print("{0}\t\t{1}".format(name,desc))

if __name__ == '__main__':
    print("RootNav 2.0")
    sys.stdout.flush()

    # Parser Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', action='store_true', default=False, help='List available models and exit')
    parser.add_argument('--model', default="wheat_bluepaper", metavar='M', help="The trained model to use (default wheat_bluepaper)")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('input_dir', type=str, help='Input directory', nargs="?")
    parser.add_argument('output_dir', type=str, help='Output directory', nargs="?")

    args = parser.parse_args()

    # If list, show and exit
    if args.list:
        list_models()
        exit()

    # Input and output directory are required
    if not args.input_dir or not args.output_dir:
        parser.print_help()
        exit()

    # Check cuda configuration and notify if cuda is unavailable but they are trying to use it
    if not torch.cuda.is_available() and not args.no_cuda:
        print ("Cuda is not available, switching to CPU")
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    # Load the model
    try:
        model_data = ModelLoader.get_model(args.model, gpu=use_cuda)
    except Exception as ex:
        print (ex)
        exit()

    # Process
    run_rootnav(model_data, use_cuda, args.input_dir, args.output_dir)
