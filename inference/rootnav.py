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
from crf import CRF


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
            model_output = model(images)[-1].data.cpu()
            model_softmax = F.softmax(model_output, dim=1)
            
            batch_count = model_output.size(0)
            if batch_count > 1:
                raise Exception("Batch size returned is greater than 1")
            
            ################################# CRF ##################################
            # Apply CRF
            mask = CRF.ApplyCRF(model_softmax.squeeze(0).numpy(),resized_img)
            enlarge(mask, realw, realh, key, output_dir)

            # Primary weighted graph
            pri_gt_mask = decode_segmap4(np.array(mask, dtype=np.uint8))
            primary_weights = distance_map(pri_gt_mask)

            # Lateral weighted graph
            lat_gt_mask = decode_segmap3(np.array(mask, dtype=np.uint8))           
            lateral_weights = distance_to_weights(lat_gt_mask)
                     
            ########################## PROCESS HEATMAPS ############################
            channel_config = model_data['channel-bindings']
            heatmap_config = model_data['channel-bindings']['heatmap']
            heatmap_index = ['Seed', 'Primary', 'Lateral']
            heatmap_output = model_output.index_select(1,torch.LongTensor([heatmap_config[i] for i in heatmap_index]))
            
            heatmap_points = {}
            for idx, binding_key in enumerate(heatmap_index):
                heatmap_points[binding_key] = nonmaximalsuppression(heatmap_output[0][idx], 0.7)

            ############################# PATH FINDING #############################
            # Filter seed and primary tip locations
            seed_locations = rrtree(heatmap_points['Seed'], 36)
            primary_tips = rrtree(heatmap_points['Primary'], 36)
            start = seed_locations[0]  
            
            lateral_goal_dict = {}

            # Create Plant structure
            plant = Plant(1, 'wheat', seed=start)

            # Search across primary roots
            for tip in primary_tips:
                path = AStar_Pri(start, tip, von_neumann_neighbors, manhattan, manhattan, primary_weights)
                if path !=[]:
                    scaled_primary_path = [(x*factor2,y*factor1) for (x,y) in path]
                    plant.roots.append(Root(scaled_primary_path, spline_tension = primary_spline_params['tension'], spline_knot_spacing = primary_spline_params['spacing']))
                    current_pid = len(plant.roots) - 1
                    for pt in path:
                        lateral_goal_dict[pt] = current_pid        

            # Filter candidate lateral root tips
            lateral_tips = rrtree(heatmap_points['Lateral'], 36)

            # Search across lateral roots
            for idxx, i in enumerate(lateral_tips):
                path, pid = AStar_Lat(i, lateral_goal_dict, von_neumann_neighbors, manhattan, manhattan, lateral_weights)
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
