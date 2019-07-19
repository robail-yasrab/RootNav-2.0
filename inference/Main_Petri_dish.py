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
from files.image_proc1 import *
from files.splines import Spline
from glob import glob
import pydensecrf.densecrf as dcrf
from files.AStar_gaps_multiplant import *
from files.AStar_gaps_laterals import *

t0 = time.time()
n_classes = 6
img_size,img_size1 = 1024, 512

def test():
    files = glob("./Petri_dish/Dataset/*.tif")
    i= len(files)
    for x in range(0, i):
        fn = files[x]
        name = []
        name = os.path.basename(fn) 
        print 'Now Reading Image: ',name
        img = misc.imread(fn)
        img007= img

        #######################RESIZE ###########################
        realh, realw= img.shape[:2]
        realw= float(realw)
        realh= float(realh)
        #####################################
        #if realh > realw:
        factor1 = realh/512
        #else:
        factor2 = (realw)/512
        factor2= float(factor2)
        ###############################################


        
        resized_img = misc.imresize(img, (img_size1, img_size1), interp='bicubic')
        
        orig_size = img.shape[:-1]
        img = misc.imresize(img, (img_size, img_size))
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()


        ################# GPU or CPU #################
        if torch.cuda.is_available():
            model.cuda(0)
            images = Variable(img.cuda(0), requires_grad=True)
        else:
            images = Variable(img, requires_grad=True)

        ##################### Process Image #################
        m = model(images)
        res1 = m[-1]
        out3= res1[:,3:4,:,:] # sec root          
        out1= res1[:,1:2,:,:] # pri root 
        out0= res1[:,0:1,:,:] # back
        
        out5= res1[:,5:6,:,:] # seed         
        out4= res1[:,4:5,:,:] # lat tip
        out2= res1[:,2:3,:,:] # pri tip    
        o = torch.cat((out2, out4, out5), 1)

        output_heatmap = o.data.cpu()
        batch_count = output_heatmap.size(0)
        channel_count = output_heatmap.size(1)
        channel_results = []
        for channel in range(0,channel_count):
            for batch in range(0,batch_count):
                prpoints = nonmaximalsuppression(output_heatmap[batch][channel], 0.9)
                channel_results.append([prpoints])  

        a4 = channel_results[2]
        a4 = np.asarray(a4) #seed tip
        a = a4.squeeze(0)

                    
        a5 = channel_results[1]
        a5 = np.asarray(a5) #lat tip
        a1 = a5.squeeze(0)
                    

        a6 = channel_results[0]
        a6 = np.asarray(a6) # pri tip
        a2 = a6.squeeze(0)
        #print 'Seed',a4.squeeze(0).shape, 'Pri_tips',a6.squeeze(0).shape, 'Lat_tip',a5.squeeze(0).shape
        ax= a6

        n = F.softmax(res1, dim=1)
        ###################### Dense CRF Processing #########################
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
            decoded_crf2 = decode_segmap2(np.array(mask, dtype=np.uint8))
            #dcrf_path = './Petri_dish/Results/'+name[:-4]+'_Color_Mask.png'
            #misc.imsave(dcrf_path, decoded_crf)

            #print("Dense CRF Processed Mask Saved at: {}".format(dcrf_path))

        pred = np.squeeze(res1.data.max(1)[1].cpu().numpy(), axis=0)
        decoded = decode_segmap1(pred) 
        decoded= np.asarray(decoded, dtype=np.float32)
        ###################################################################


        ########################### Build R Tree ####################################
        trees = []
        output_points= np.asarray(ax,  dtype = np.float32)

        for c in range(len(output_points)):
            idx = index.Index(interleaved=False)
            distance_threshold = 16 # 8^2
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
        pgoals = channel_points[0:h].astype(int)
        pgoals = tuple(map(tuple, pgoals))
        ###################### Rtree for seed ##########################
        pstart = rrtree(a4)
        pstart = pstart.astype(int)   
        pstart = tuple(map(tuple, pstart))

        
        idx= len(pgoals)
        distance = manhattan
        heuristic = manhattan
        pri_gt_mask = decode_segmap3(np.array(mask, dtype=np.uint8))
        weights = distance_map(pri_gt_mask) #distance map



        points = []
        my_list1 = []
        my_list2 = []
        my_list8 = []
        my_list9 = []
        my_list11 = []
        my_list12 = []
        pstart = list(pstart)
        pgoalss = list(pgoals)
        pgoals = {x: i for i, x in enumerate(pgoalss)}




        plant_primary_roots = {}

        for  idx, val in enumerate(pstart):
        #for idx, i in enumerate(pgoals):
            #idx= pgoals.index(i)
            my_list1.append([])
            my_list2.append([])
            my_list11.append([])
            my_list12.append([])
            path, plant_id = AStar_pri_m(val, pgoals, von_neumann_neighbors, distance, heuristic, weights)
            if path !=[]:
                path = list(path)
                path.reverse()                   
                if plant_id not in plant_primary_roots:           
                    plant_primary_roots[plant_id] = [path]
                else:
                    plant_primary_roots[plant_id].append(path)

                for position in path:
                    x,y = position
                    my_list11[idx].append(x)
                    my_list12[idx].append(y)
                    x,y = float(x*factor2), float(y*factor2)
                    my_list1[idx].append(x)
                    my_list2[idx].append(y)
        Lat_goal = []           
        for p,q in zip(my_list11, my_list12):
            if p!=[] and q!=[]:
                for x, y in zip(p, q):
                    Lat_goal.append((x,y))




        lat_gt_mask = decode_segmap4(np.array(mask, dtype=np.uint8))
        weights = distance_to_weights(lat_gt_mask) #distance map 
        misc.imsave("q.png", weights) 


        #img2= ext_color(decoded_crf2, my_list11, my_list12) 

        my_list4 = []
        my_list5 = []
        my_list6 = []
        my_list7 = []
        goal = rrtree(a5)
        goal = goal.astype(int)   
        goal = tuple(map(tuple, goal))
        start = [(250, 46)]
        for idxx, i in enumerate(goal):
            my_list4.append([])
            my_list5.append([])
            path = AStar_Lat(i, Lat_goal, von_neumann_neighbors, distance, heuristic, weights)
            #path = AStar2(i, start, von_neumann_neighborsA, distance, heuristic, img2)
            if path !=[]:
                path = list(path)
                path.reverse()
                for position in path:
	                x,y = position
	                x,y = float((x)*factor2), float(y*factor2)
	                my_list4[idxx].append(x)
	                my_list5[idxx].append(y)
	                my_list6.append(x)
	                my_list7.append(y)
	                
	    root = ET.Element('rsml')
	    metadata = ET.SubElement(root, 'metadata')
	    ET.SubElement(metadata, 'version').text = "1"
	    ET.SubElement(metadata, 'unit').text = "pixel"
	    ET.SubElement(metadata, 'resolution').text = "1"
	    ET.SubElement(metadata, 'last-modified').text = "1"
	    ET.SubElement(metadata, 'software').text = "ROOT_NAV.2.0"
	    ET.SubElement(metadata, 'user').text = "Robi"
	    ET.SubElement(metadata, 'file-key').text = name[:-4]
	    scene = ET.SubElement(root, 'scene')
        count_lat = []
        for l,m in zip(my_list1, my_list2):
            idx= my_list1.index(l)
            c = np.array(l)
            d = np.array(m)
            if c!=[] and d!=[]:
            	plant = ET.SubElement(scene, 'plant', id= str(idx), label="simple_arabidopsis_rsa")
                priroot = ET.SubElement(plant, 'root', id=str(idx), label="primary", poaccession="1")
                geometry = ET.SubElement(priroot, 'geometry')
                polyline = ET.SubElement(geometry, 'polyline')
                rootnavspline = ET.SubElement(geometry, 'rootnavspline', controlpointseparation= "50", tension="0.5")
                points = []
                for a, b in zip(c, d):
                    points.append((a, b))
                    point = ET.SubElement(polyline, 'point', x=str(a), y=str(b))
                if points != []:
	            	s = Spline(points, tension = 0.5, knot_spacing = 50)
	            	poly = s.polyline(sample_spacing = 1)
	            	for c in s.knots:
	            		point = ET.SubElement(rootnavspline, 'point', x=str(c[0]), y=str(c[1]))
	            		my_list8.append(c[0])
	            		my_list9.append(c[1])
	       ################################### Latral Root ##################################
            for g,h in zip(l,m):
                x, y = g, h
                neighbors = [(x, y)]
                for s,t in zip(my_list4, my_list5):
                    id_lat= my_list4.index(s)
                    for p in neighbors:
                        s= map(int, s)
                        t= map(int, t)
                        #if int(p[0]) in s and int(p[1]) in t and id_lat not in count_lat:
                        if int(p[0]) in s and int(p[1]) in t and id_lat not in count_lat:
                            count_lat.append(id_lat)
                            #print id_lat, count_lat
                            idxxx= str(idx)+"."+str(id_lat)
                            secroot = ET.SubElement(priroot, 'root', ID=idxxx, label="lat")
                            geometry = ET.SubElement(secroot, 'geometry')
                            polyline = ET.SubElement(geometry, 'polyline')
                            rootnavspline = ET.SubElement(geometry, 'rootnavspline', controlpointseparation= "20", tension="0.5")
                            pointss = [] 
                            for n, p in zip(s, t):
                                pointss.append((n, p))
                                n = np.array(n)
                                p = np.array(p)
                                point = ET.SubElement(polyline, 'point', x=str(n), y=str(p))
                            if pointss!=[]:
                            	ss = Spline(pointss, tension = 0.5, knot_spacing = 20)
                            	polyy = ss.polyline(sample_spacing = 1)
                            	for cc in  ss.knots:
                                	point = ET.SubElement(rootnavspline, 'point', x=str(cc[0]), y=str(cc[1]))




        plt.figure()
        plt.imshow(img007)
        plt.scatter(my_list8, my_list9, s=1, marker='.', c='b')
        plt.scatter(my_list6, my_list7, s=1, marker='.', c='r')
        plt.show()
        plt.axis('off')
        plt.savefig('./Petri_dish/Results/'+name[:-4]+'_Spline.png')
        plt.clf()
        plt.cla()
        plt.close()
  
        tree = ET.ElementTree(root)
        output_file = open('./Petri_dish/Results/'+name[:-4]+'.rsml', 'w')
        output_file.write( prettify(root))
        output_file.close()

        #################################################
    print("Dense CRF Post Processed Mask and RSML File is Saved at: Petri_dish/Results/")
    t1 = time.time()
    total = t1-t0
    print ("RSML Time elapsed:", total, "\n")

if __name__ == '__main__':
    # Setup Model
    model_path = "./models/arabidopsis_irplate.pkl"
    model = hg()
    state = convert_state_dict(torch.load(model_path)['model_state'])
    model.load_state_dict(state)
    model.eval()
    test()
