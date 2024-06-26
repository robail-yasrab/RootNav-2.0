import time, sys, os, argparse
import torch
from torch.nn.functional import softmax
import numpy as np
from PIL import Image
from PIL.Image import BICUBIC
from torch.autograd import Variable
from utils import nonmaximalsuppression as nms, rrtree
from utils import image_output, distance_map, distance_to_weights
from astar import AStar_Pri, AStar_Lat, von_neumann_neighbors, manhattan
from glob import glob
from rsml import RSMLWriter, Plant, Root
from models import ModelLoader
from crf import CRF
import logging

n_classes = 6
fileExtensions = set([".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP" ])

logger = logging.getLogger()

def run_rootnav(model_data, use_cuda, args, input_dir, output_dir):

    # Load parameters
    model = model_data['model']

    pathing_config = model_data['configuration']['pathing']    
    primary_spline_params = pathing_config['spline-config']['primary']
    lateral_spline_params = pathing_config['spline-config']['lateral']
  
    net_config = model_data['configuration']['network']
    heatmap_config = net_config['channel-bindings']['heatmap']
    segmap_config = net_config['channel-bindings']['segmentation']

    net_input_size = net_config['input-size']
    net_output_size = net_config['output-size']
    normalisation_scale = net_config['scale']

    segmentation_images = args.segmentation_images

    files = glob(os.path.join(input_dir, "*.*"))

    logger.debug(f"Found {len(files)} file in input directory")

    for file in files:
        extension = os.path.splitext(file)[1].upper()
        if extension in fileExtensions:
            t0 = time.time()
            name = os.path.basename(file) 
            key = os.path.splitext(name)[0]

            logger.info('Now processing {0}'.format(name))
            sys.stdout.flush()

            pil_img = Image.open(file)

            logger.debug(f"Read image, size {pil_img.size}, format {pil_img.mode}")

            if (pil_img.mode != "RGB"):
                pil_img = pil_img.convert("RGB")
                logger.debug(f"Converting image to RGB")

            ####################### RESIZE #########################################
            realw, realh = pil_img.size
            realw = float(realw)
            realh = float(realh)
            factor1 = realh / 512.0
            factor2 = realw / 512.0

            resized_img = np.array(pil_img.resize((net_output_size, net_output_size),resample=BICUBIC))
            img = np.array(pil_img.resize((net_input_size, net_input_size)))

            logger.debug(f"Resizing image to {net_input_size}x{net_input_size}")  

            ########################### IMAGE PREP #################################
            if normalisation_scale != 1:
                img = img.astype(float) * normalisation_scale
            
            # NHWC -> NCHW
            img = np.transpose(img,(2, 0, 1))
            img = np.expand_dims(img, 0)
            img = torch.from_numpy(img).float()

            if use_cuda:
                model.cuda(0)
                images = Variable(img.cuda(0), requires_grad=False)
            else:
                images = Variable(img, requires_grad=False)

            ######################## MODEL FORWARD #################################

            logging.debug("Processing image through network")
            model_output = model(images)[-1].data.cpu()
            model_softmax = softmax(model_output, dim=1)
            
            batch_count = model_output.size(0)
            if batch_count > 1:
                raise Exception("Batch size returned is greater than 1")
            
            ################################# CRF ##################################
            # Apply CRF
            mask = CRF.ApplyCRF(model_softmax.squeeze(0), resized_img)

            # Primary weighted graph
            pri_gt_mask = CRF.decode_channel(mask, [segmap_config['Primary'],heatmap_config['Seed']])
            primary_weights = distance_map(pri_gt_mask)

            # Lateral weighted graph
            lat_gt_mask = CRF.decode_channel(mask, segmap_config['Lateral'])         
            lateral_weights = distance_to_weights(lat_gt_mask)
                     
            ########################## PROCESS HEATMAPS ############################
            logging.debug("Processing output heatmaps")
            heatmap_index = ['Seed', 'Primary', 'Lateral']
            heatmap_output = model_output.index_select(1,torch.LongTensor([heatmap_config[i] for i in heatmap_index]))
            
            heatmap_points = {}
            for idx, binding_key in enumerate(heatmap_index):
                heatmap_points[binding_key] = nms(heatmap_output[0][idx], 0.7)

            ############################# PATH FINDING #############################
            # Filter seed and primary tip locations
            seed_locations = rrtree(heatmap_points['Seed'], pathing_config['rtree-threshold'])
            primary_tips = rrtree(heatmap_points['Primary'], pathing_config['rtree-threshold'])


            if len(seed_locations) < 1:
                logger.warning("No seed locations found - no output")
                continue
            else:
                logger.debug(f"Found {len(heatmap_points['Seed'])} raw seed locations, filtered to {len(seed_locations)} location{'' if len(seed_locations) == 1 else 's'}")

            if len(primary_tips) < 1:
                logger.warning("No first order roots found - no output")
                continue
            else:
                logger.debug(f"Found {len(heatmap_points['Primary'])} raw first order root tip locations, filtered to {len(primary_tips)} location{'' if len(primary_tips) == 1 else 's'}")

            primary_goal_dict = {pt:ix for ix,pt in enumerate(seed_locations)}
            lateral_goal_dict = {}

            # Create Plant structure
            plants = [Plant(ix, 'wheat', seed=pt) for ix,pt in enumerate(seed_locations)]
            primary_root_index = []

            # Search across primary roots
            for tip in primary_tips:
                path,plant_id = AStar_Pri(tip, primary_goal_dict, von_neumann_neighbors, manhattan, primary_weights, pathing_config['max-primary-distance'])
                if path !=[]:
                    scaled_primary_path = [(x*factor2,y*factor1) for (x,y) in reversed(path)]
                    primary_root = Root(scaled_primary_path, spline_tension = primary_spline_params['tension'], spline_knot_spacing = primary_spline_params['spacing'])
                    plants[plant_id].roots.append(primary_root)
                    primary_root_index.append(primary_root)
                    current_pid = len(primary_root_index) - 1
                    for pt in path:
                        lateral_goal_dict[pt] = current_pid
                else:
                    logger.debug(f"Removed primary root starting at pixel location {(round(tip[0] * factor1), round(tip[1] * factor2))} as no valid path was found")

            # Filter candidate lateral root tips
            lateral_tips = rrtree(heatmap_points['Lateral'], pathing_config['rtree-threshold'])
            logger.debug(f"Found {len(heatmap_points['Lateral'])} raw first order root tip locations, filtered to {len(lateral_tips)} location{'' if len(lateral_tips) == 1 else 's'}")

            # Search across lateral roots
            for idxx, i in enumerate(lateral_tips):
                path, pid = AStar_Lat(i, lateral_goal_dict, von_neumann_neighbors, lateral_weights, pathing_config['max-lateral-distance'])
                if path !=[]:
                    scaled_lateral_path = [(x*factor2,y*factor1) for (x,y) in reversed(path)]
                    lateral_root = Root(scaled_lateral_path, spline_tension = lateral_spline_params['tension'], spline_knot_spacing = lateral_spline_params['spacing'])
                    primary_root_index[pid].roots.append(lateral_root)
                else:
                    logger.debug(f"Removed second order root starting at pixel location {(round(i[0] * factor1), round(i[1] * factor2))} as no valid path was found")

            # Filter plants with no roots (E.g. incorrect seed location)
            plant_count = len(plants)
            plants = [plant for plant in plants if plant.roots is not None and len(plant.roots) > 0]

            if (plant_count != len(plants)):
                logging.debug(f"Removed {plant_count - len(plants)} plant{'' if plant_count - len(plants) == 1 else 's'} with no roots found")

            if len(plants) < 1:
                # No viable primary roots found for any plant
                logger.warning("No valid paths found between tips and seed locations - no output")
                continue
            
            # Output to RSML
            RSMLWriter.save(key, output_dir, plants)

            # Output images
            image_output(mask, realw, realh, key, net_config['channel-bindings'], output_dir, segmentation_images)

            ############################# Total time per Image ######################
            logger.info(f"RSML and mask for {len(plants)} plant{'' if len(plants) == 1 else 's'} output saved in: {output_dir}")
            t1 = time.time()
            total = t1-t0
            logger.info("Time elapsed: {0:.2f}s\n".format(total))

def print_table(table):
    col_width = [max(len(x) for x in col) for col in zip(*table)]
    hline = "-" * (sum(w for w in col_width) + 7)
    print (hline)
    for i,line in enumerate(table):
        print ("| " + " | ".join("{0:{1}}".format(x, col_width[i])
                                for i, x in enumerate(line)) + " |")
        if i == 0:
            print (hline)
    print (hline)

class list_action(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        table_data = [("Model", "Description")] + [(name,desc) for name, desc in ModelLoader.list_models(True)]
        print_table(table_data)
        exit()

class info_action(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        data = ModelLoader.model_info(name=values[0])
        if data is not None:
            left_col_width = max([len(t[0]) for t in data])
            for (k,v) in data:
                if not isinstance(v, list):
                    print ("{1:>{0}}: {2}".format(left_col_width, k, v))
                else:
                    if (len(v) <= 1):
                        continue
                        
                    print ("{1:>{0}}: {2}".format(left_col_width, k, v[0]))
                    for i in range(1,len(v)):
                        print ("{1:>{0}}  {2}".format(left_col_width, "", v[i])) 
        exit()
