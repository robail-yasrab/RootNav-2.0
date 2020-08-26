import os
import sys
import yaml
import time
import shutil
import torch
import random
import argparse
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import scipy.misc as misc
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils import data
from tqdm import tqdm
from rootnav2.hourglass import hg
import cv2
from rootnav2.loader import get_loader 
from rootnav2.utils import get_logger
from rootnav2.metrics import runningScore, averageMeter, LocalisationAccuracyMeter
from pathlib import Path
from tensorboardX import SummaryWriter
from rootnav2.utils import convert_state_dict, decode_segmap, dict_collate
from rootnav2.accuracy import nonmaximalsuppression as nms, rrtree, evaluate_points
import collections
import math

def test(args):
    # Load Config
    with open(args.config) as fp:
        cfg = yaml.load(fp)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    print ("Dataset Loading from", data_path)

    t_loader = data_loader(
        data_path,
        split='test')

    n_classes = t_loader.n_classes
    testloader = data.DataLoader(t_loader,
                                batch_size=cfg['training']['batch_size'], 
                                num_workers=cfg['training']['n_workers'],
                                collate_fn = dict_collate)

    # Setup Metrics
    running_metrics = runningScore(n_classes)

    # Setup Model
    print ("Loading model weights")
    model =  hg()
    state = convert_state_dict(torch.load(args.model)["model_state"])
    model.load_state_dict(state)
    
    model = torch.nn.DataParallel(model)
    model.to(device)

    model.eval()

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    print ("Starting test")
    
    accuracy_meter = LocalisationAccuracyMeter(3) # Seed, Primary, Lateral
    total = 0

    with torch.no_grad():
        for i_val, (images, masks, annotations) in enumerate(testloader):
            images = images.to(device)
            outputs = model(images)[-1].data.cpu()
            total += outputs.size(0)

            # from torchvision.utils import save_image
            # save_image(images[0], "source.png")
            # #save_image(outputs[0], 'seed_hm.png')
            # #save_image(outputs[1], 'pri_hm.png')
            # #save_image(outputs[2], 'lat_hm.png')
            # save_image(decode_segmap(outputs.cpu().argmax(1)[0]), 'pred.png')
            # save_image(decode_segmap(masks[0]), 'mask.png')
            # exit()

            ##### SEGMENTATION ACCURACY #####
            # Class predictions
            pred = outputs.cpu().argmax(1).numpy()
            gt = masks.numpy()
            # Update metrics
            running_metrics.update(gt, pred)
            #################################

            # 0 BG
            # 1 Lat seg
            # 2 Lat tip
            # 3 Pri seg
            # 4 Pri tip
            # 5 Seed location

            ##### LOCALISATION ACCURACY #####
            # For each channel / gt annotation
            # Indexes output, tensor and dict together
            channel_results = []
            for idx, (channel_idx, annotation_idx) in enumerate(zip([5, 4, 2], ["seeds", "primary", "lateral"])):
                ctp, cfp, cfn = 0, 0, 0

                # For each image in the batch
                for batch_idx in range(outputs.size(0)):
                    current = outputs[batch_idx]
                    pred = rrtree(nms(current[channel_idx], 0.7), 36)
                    gt = annotations[annotation_idx][batch_idx]
                    
                    # Accuracy of current image on current channel
                    tp, fp, fn = evaluate_points(pred, gt, 6)

                    # Accumulate results    
                    ctp += tp
                    cfp += fp
                    cfn += fn
                
                channel_results.append([ctp, cfp, cfn])

            accuracy_meter.update(channel_results)
            #################################

        print ("Processed {0} images".format(total))
        metrics, class_iou = running_metrics.get_scores()

        print ("\r\nSegmentation Results:")
        print ("Overall Accuracy : {0:.4f}".format(metrics['Overall Acc: \t']))
        print ("Mean Accuracy : {0:.4f}".format(metrics['Mean Acc : \t']))
        print ("FreqW Accuracy : {0:.4f}".format(metrics['FreqW Acc : \t']))
        print ("Root Mean IoU: {0:.4f}".format((class_iou[0] + class_iou[1] + class_iou[3]) / 3))

        print ("\r\nLocalisation Results:")
        format_string = "{0}\tPrecision: {1:.4f}  Recall: {2:.4f}  F1: {3:.4f}"

        for (channel, results) in zip(["Seeds", "Primary", "Lateral"], accuracy_meter.f1()):
            print (format_string.format(channel, *results))
        
        print("\r\nTest completed succesfully")
