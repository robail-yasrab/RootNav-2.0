import time
import sys, os
import torch
import argparse
import numpy as np
import scipy.misc as misc
from torch.autograd import Variable
from files.func import *
from files.astar import *
from files.rrtree import rrtree
from files.image_proc import *
from files.AStar_gaps import *
from files.AStar_gaps_laterals import *
from glob import glob
from rsml import RSMLWriter, Plant, Root
from models import ModelLoader
from crf import CRF


n_classes = 6
fileExtensions = set([".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP" ])

def run_rootnav(model_data, use_cuda, use_crf, input_dir, output_dir, no_segmentation_images):
    
    # Load parameters
    model = model_data['model']
    multi_plant = model_data['multi-plant']
    primary_spline_params = model_data['pathing-config']['spline-config']['primary']
    lateral_spline_params = model_data['pathing-config']['spline-config']['lateral']
  
    net_config = model_data['net-config']
    heatmap_config = net_config['channel-bindings']['heatmap']
    segmap_config = net_config['channel-bindings']['segmentation']

    net_input_size = net_config['input-size']
    net_output_size = net_config['output-size']
    normalisation_scale = net_config['scale']

    pathing_config = model_data['pathing-config']

    files = glob(os.path.join(input_dir, "*.*"))
    for file in files:
        extension = os.path.splitext(file)[1].upper()
        if extension in fileExtensions:
            t0 = time.time()
            name = os.path.basename(file) 
            key = os.path.splitext(name)[0]

            print ('Now Reading {0}'.format(name))
            sys.stdout.flush()
            img = misc.imread(file)

            ####################### RESIZE #########################################
            realh, realw = img.shape[:2]
            realw = float(realw)
            realh = float(realh)
            factor1 = realh / 512.0
            factor2 = realw / 512.0

            resized_img = misc.imresize(img, (net_output_size, net_output_size), interp='bicubic')
            orig_size = img.shape[:-1]

            img = misc.imresize(img, (net_input_size, net_input_size))

            ########################### IMAGE PREP #################################
            if normalisation_scale != 1:
                img = img.astype(float) * normalisation_scale
            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()

            if use_cuda:
                model.cuda(0)
                images = Variable(img.cuda(0), requires_grad=False)
            else:
                images = Variable(img, requires_grad=False)

            ######################## MODEL FORWARD #################################
            model_output = model(images)[-1].data.cpu()
            model_softmax = F.softmax(model_output, dim=1)
            
            batch_count = model_output.size(0)
            if batch_count > 1:
                raise Exception("Batch size returned is greater than 1")
            
            ################################# CRF ##################################
            # Apply CRF
            mask = CRF.ApplyCRF(model_softmax.squeeze(0), resized_img, use_crf)
            enlarge(mask, realw, realh, key, net_config['channel-bindings'], output_dir, no_segmentation_images)
            
            # Primary weighted graph
            pri_gt_mask = CRF.decode_channel(mask, [segmap_config['Primary'],heatmap_config['Seed']])
            primary_weights = distance_map(pri_gt_mask)

            # Lateral weighted graph
            lat_gt_mask = CRF.decode_channel(mask, segmap_config['Lateral'])         
            lateral_weights = distance_to_weights(lat_gt_mask)
                     
            ########################## PROCESS HEATMAPS ############################
            heatmap_index = ['Seed', 'Primary', 'Lateral']
            heatmap_output = model_output.index_select(1,torch.LongTensor([heatmap_config[i] for i in heatmap_index]))
            
            heatmap_points = {}
            for idx, binding_key in enumerate(heatmap_index):
                heatmap_points[binding_key] = nonmaximalsuppression(heatmap_output[0][idx], 0.7)

            ############################# PATH FINDING #############################
            # Filter seed and primary tip locations
            seed_locations = rrtree(heatmap_points['Seed'], pathing_config['rtree-threshold'])
            primary_tips = rrtree(heatmap_points['Primary'], pathing_config['rtree-threshold'])

            primary_goal_dict = {pt:ix for ix,pt in enumerate(seed_locations)}
            lateral_goal_dict = {}

            # Create Plant structure
            plants = [Plant(ix, 'wheat', seed=pt) for ix,pt in enumerate(seed_locations)]
            primary_root_index = []

            # Search across primary roots
            for tip in primary_tips:
                path,plant_id = AStar_Pri(tip, primary_goal_dict, von_neumann_neighbors, manhattan, manhattan, primary_weights, pathing_config['max-primary-distance'])
                if path !=[]:
                    scaled_primary_path = [(x*factor2,y*factor1) for (x,y) in reversed(path)]
                    primary_root = Root(scaled_primary_path, spline_tension = primary_spline_params['tension'], spline_knot_spacing = primary_spline_params['spacing'])
                    plants[plant_id].roots.append(primary_root)
                    primary_root_index.append(primary_root)
                    current_pid = len(primary_root_index) - 1
                    for pt in path:
                        lateral_goal_dict[pt] = current_pid        

            # Filter candidate lateral root tips
            lateral_tips = rrtree(heatmap_points['Lateral'], pathing_config['rtree-threshold'])

            # Search across lateral roots
            for idxx, i in enumerate(lateral_tips):
                path, pid = AStar_Lat(i, lateral_goal_dict, von_neumann_neighbors, manhattan, manhattan, lateral_weights, pathing_config['max-primary-distance'])
                if path !=[]:
                    scaled_lateral_path = [(x*factor2,y*factor1) for (x,y) in reversed(path)]
                    lateral_root = Root(scaled_lateral_path, spline_tension = lateral_spline_params['tension'], spline_knot_spacing = lateral_spline_params['spacing'])
                    primary_root_index[pid].roots.append(lateral_root)

            # Filter plants with no roots (E.g. incorrect seed location)
            plants = [plant for plant in plants if plant.roots is not None and len(plant.roots) > 0]

            # Output to RSML
            RSMLWriter.save(key, output_dir, plants)

            ############################# Total time per Image ######################
            print("RSML and mask output saved in: {0}".format(output_dir))
            t1 = time.time()
            total = t1-t0
            print ("Time elapsed: {0:.2f}s\n".format(total))

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
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--no_crf', action='store_true', default=False, help='disables CRF post-processing')
    parser.add_argument('input_dir', type=str, help='Input directory', nargs="?")
    parser.add_argument('output_dir', type=str, help='Output directory', nargs="?")
    parser.add_argument('--no_segmentation_images', action='store_true', default=False, help='Reduce output files to minimum')

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

    # Check CRF flag
    use_crf = not args.no_crf

    # Load the model
    try:
        model_data = ModelLoader.get_model(args.model, gpu=use_cuda)
    except Exception as ex:
        print (ex)
        exit()

    # Process
    run_rootnav(model_data, use_cuda, use_crf, args.input_dir, args.output_dir, args.no_segmentation_images)
