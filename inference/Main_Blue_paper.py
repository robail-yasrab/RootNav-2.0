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
from rtree import index
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
from rsml import Spline, Plant as P, Root as R, RSMLWriter


n_classes = 6
img_size,img_size1 = 1024, 512
t0 = time.time()
fileExtensions = [ "jpg", "JPG", "png", "tif" ]

def test():
    for extension in fileExtensions:
        files = glob("./Blue_paper/Dataset/*."+extension)
        i= len(files)
        #print i
        for x in range(0, i):
            fn = files[x]
            name = os.path.basename(fn) 
            print 'Now Reading',name
            sys.stdout.flush()
            img = misc.imread(fn)
            img007= img

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


            if torch.cuda.is_available():
                model.cuda(0)
                images = Variable(img.cuda(0), requires_grad=True)
            else:
                images = Variable(img, requires_grad=True)


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
            a6 = np.asarray(a6) # latrl tip
            a2 = a6.squeeze(0)

            


            n = F.softmax(res1, dim=1)

            if 1==1:
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
                enlarge(mask, realw, realh, name)
                decoded_crf = decode_segmap(np.array(mask, dtype=np.uint8))

            pred = np.squeeze(res1.data.max(1)[1].cpu().numpy(), axis=0)
            decoded = decode_segmap1(pred) 
            #misc.imsave('11.png', decoded)
            decoded= np.asarray(decoded, dtype=np.float32)
            ###################################################################

            #white = ext_white_mask(decoded_crf)
            ######################## COLOR GT #################################

        ######################## primery root GT ###########################


            trees = []
            if a4!=[]:
                ax= np.concatenate((a4, a5), axis=1)
                output_points= np.asarray(ax,  dtype = np.float32)
            else:
                output_points= np.asarray(a5,  dtype = np.float32)
            
            


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

            counts = []
            channel_points=[]
            for c in range(len(output_points)):
                channel_points = [output_points[c][pt] for pt in trees[c].intersection((-1,512,-1,512))]
            channel_points = np.asarray(channel_points)
            

            (h, w)= channel_points.shape  
            if a4!=[]:           
                start = channel_points[0].astype(int)
                goals = channel_points[1:h].astype(int)
                start = tuple(start)
            else:
                start =(218, 41)
                goals = channel_points.astype(int)

            
            goals = tuple(map(tuple, goals))
            
            ###########################RSML################################
            root = ET.Element('rsml') 
            metadata = ET.SubElement(root, 'metadata')
            ET.SubElement(metadata,  'version').text = "1"
            ET.SubElement(metadata, 'unit').text = "pixel"
            ET.SubElement(metadata, 'resolution').text = "1"
            ET.SubElement(metadata, 'last-modified').text = "1"
            ET.SubElement(metadata, 'software').text = "ROOT_NAV.2.0"
            ET.SubElement(metadata, 'user').text = "Robi"
            ET.SubElement(metadata, 'file-key').text = name[:-4]
            scene = ET.SubElement(root, 'scene')
            plant = ET.SubElement(scene, 'plant', id= "1", label="simple_arabidopsis_rsa")
            ###############################################################
            idx= len(goals)
         
            decoded = np.asarray(decoded, dtype = np.float32)
            gray_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2GRAY)
            gray_image = Image.fromarray(gray_image)
            path_pixels = gray_image.load()
            distance = manhattan
            heuristic = manhattan
            pri_gt_mask = decode_segmap4(np.array(mask, dtype=np.uint8))
            weights = distance_map(pri_gt_mask) #distance map
            
            primary_root_paths = []

            for idx, i in enumerate(goals):
                path = AStar_Pri(start, i, von_neumann_neighbors, distance, heuristic, weights)
                if path !=[]:
                    primary_root_paths.append(path)

            lateral_goals = [(idx,position) for idx, primary in enumerate(primary_root_paths) for position in primary]
            lateral_goal_dict = {}
            for idx, position in lateral_goals:
                if position not in lateral_goal_dict:
                    lateral_goal_dict[position] = idx

            lat_gt_mask = decode_segmap3(np.array(mask, dtype=np.uint8))           
            img3= distance_to_weights(lat_gt_mask)            

            lateral_tips = rrtree(a6)
            lateral_tips = lateral_tips.astype(int)   
            lateral_tips = tuple(map(tuple, lateral_tips))

            lateral_root_paths = x = [[] for i in range(len(primary_root_paths))]

            for idxx, i in enumerate(lateral_tips):
                path, pid = AStar_Lat(i, lateral_goal_dict, von_neumann_neighbors, distance, heuristic, img3)
                if path !=[]:
                    lateral_root_paths[pid].append(list(reversed(path)))
                else:
                    pass

            # Create Plant structure
            plant = P(1, 'wheat', seed=primary_root_paths[0][0], roots = [])
            for idx, primary_root_path in enumerate(primary_root_paths):
                l_roots = [R([(x*factor2,y*factor1) for (x,y) in path], roots = None, spline_tension = 0.5, spline_knot_spacing = 40) for path in lateral_root_paths[idx]]
                scaled_primary_path = [(x*factor2,y*factor1) for (x,y) in primary_root_path]
                p_root = R(scaled_primary_path, roots = l_roots, spline_tension = 0.5, spline_knot_spacing = 100)
                plant.roots.append(p_root)

            rsml_text = RSMLWriter.save(name[:-4], [plant])

            output_file = open('./Blue_paper/Results/'+name[:-4]+'.rsml', 'w')
            output_file.write(rsml_text)
            output_file.close()

            #############################Totel time per Image ######################
            print("Dense CRF Post Processed Mask and RSML File is Saved at: Blue_paper/Results/")
            t1 = time.time()
            total = t1-t0
            print ("RSML Time elapsed:", total, "\n")

if __name__ == '__main__':
    # Setup Model
    #model_path = "./Model/Single_blue.pkl"
    model_path = "./models/wheat_bluepaper.pkl"
    model = hg()
    state = convert_state_dict(torch.load(model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    test()
