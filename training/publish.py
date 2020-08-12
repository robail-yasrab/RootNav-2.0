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
from rootnav2.n_hourglass import hg
import cv2
from rootnav2.loss import get_loss_function
from rootnav2.loader import get_loader 
from rootnav2.utils import get_logger
from rootnav2.metrics import runningScore, averageMeter
from rootnav2.schedulers import get_scheduler
from rootnav2.optimizers import get_optimizer
from pathlib import Path
import json
from glob import glob
import uuid
import hashlib

PREFIX_LENGTH = 8

def find_model(path, name):
        # Scan model Directory
        files = glob("{0}/*.json".format(path))
        
        model_list = []
        for file in files:
            with open(file, 'r') as f:
                model_json = json.loads(f.read())
                if model_json['name'] == name:
                    return model_json
        return None

def create_blank_model():
    return {
        "name": "",
        "description": "",
        "uuid": str(uuid.uuid4()),
        "history": {
            "model": {
                "parent-model": "",
                "trained-by": {
                    "fullname": "Author full name",
                    "affiliation": "Author affiliation",
                    "email": "Author email"
                },
                "license": "Trained network license url"
            },
            "dataset": {
                "owner": [
                    {
                        "fullname": "Dataset owner 1",
                        "affiliation": "Owner 1 affiliation",
                        "email": "Owner 1 email"
                    },
                    {
                        "fullname": "Dataset owner 2",
                        "affiliation": "Owner 2 affiliation",
                        "email": "Owner 2 email"
                    }
                ],
                "url": "Dataset location url",
                "license": "Dataset license url"
            }
        },
        "configuration": {
            "multi-plant": False,
            "network": {
                "architecture": "hg",
                "weights": "",
                "url":"",
                "scale": 0.00392156862745098,
                "input-size": 1024,
                "output-size": 512,
                "channel-bindings": {
                    "segmentation": { "Background": 0, "Primary": 1, "Lateral": 2 },
                    "heatmap": { "Seed": 3, "Primary": 4, "Lateral": 5 }
                }
            },
            "pathing": {
                "rtree-threshold": 36,
                "nms-threshold": 0.7,
                "max-primary-distance": 400,
                "max-lateral-distance": 200,
                "spline-config": {
                    "primary": { "tension": 0.5, "spacing": 50 },
                    "lateral": { "tension": 0.5, "spacing": 20 }
                }
            }
        }
    }


def publish(args):
    print ("Publish")
    print (args)

    model_path = Path(args.model).resolve()

    publish_folder = Path(args.output_dir).resolve()
    publish_json_path = publish_folder / "{0}.json".format(args.name)

    # If published json already exist, user should confirm an overwrite is ok
    if os.path.isfile(publish_json_path):
        response = input("Published JSON already exists, continue anyway? [y/n] ")
        if response.lower() != 'y':
            print ("Exiting.")
            exit()
    
    # Validate parent and obtain uuid
    if args.parent != None:
        # Resolve inference model directory
        inference_model_folder = Path(__file__).resolve().parent.parent / 'inference' / 'models'
        
        # Load parent json
        parent_json = find_model(inference_model_folder, args.parent)

        if parent_json is None:
            print ("Named parent could not be located, exiting.")
            exit()
    else:
        parent_json = None

    parent_uuid = parent_json["uuid"] if parent_json is not None else ""

    # Create new model json description
    model_json = create_blank_model()
    model_json["name"] = args.name
    model_json["history"]["model"]["parent-model"] = parent_uuid
    model_json["configuration"]["multi-plant"] = args.multi_plant

    digest = hashlib.sha256()

    # Calculate hash prefix for this model
    with open(model_path, 'rb') as model_f:
        data = model_f.read(1024)
        while len(data) > 0:
            digest.update(data)
            data = model_f.read(1024)
    
    hash_prefix = digest.hexdigest()[:PREFIX_LENGTH]

    # Model install directory
    model_json["configuration"]["network"]["weights"] = "{0}-{1}.pth".format(args.name, hash_prefix)
    publish_model_path = publish_folder / model_json["configuration"]["network"]["weights"]

    print (model_json)

    """
    if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
    """


    """
    "name": "osr_bluepaper",
    "description": "Rapeseed grown on blue paper",
    "uuid": "5edd7fdc-ceb1-44fa-9dfd-ef1197abac48",
    "history": {
        "model": {
            "parent-model": "2d9ae372-b128-43de-87de-a905ac420b08",
            "trained-by": {
                "fullname": "Robail Yasrab",
                "affiliation": "University of Nottingham",
                "email": "robail.yasrab@nottingham.ac.uk"
            },
            "license": "https://creativecommons.org/licenses/by/4.0/"
        },
        "dataset": {
            "owner": [
                {
                    "fullname": "Jonathan Atkinson",
                    "affiliation": "University of Nottingham",
                    "email": "jonathan.atkinson@nottingham.ac.uk"
                },
                {
                    "fullname": "Darren Wells",
                    "affiliation": "University of Nottingham",
                    "email": "darren.wells@nottingham.ac.uk"
                }
            ],
            "url": "https://plantimages.nottingham.ac.uk",
            "license": "https://creativecommons.org/licenses/by-nc/4.0/"
        }
    },
    "configuration": {
        "multi-plant": false,
        "network": {
            "architecture": "hg",
            "weights": "osr_bluepaper.pth",
            "url":"https://cvl.cs.nott.ac.uk/resources/trainedmodels/osr_bluepaper-083ed788.pth",
            "scale": 0.00392156862745098,
            "input-size": 1024,
            "output-size": 512,
            "channel-bindings": {
                "segmentation": { "Background": 0, "Primary": 3, "Lateral": 1 },
                "heatmap": { "Seed": 5, "Primary": 4, "Lateral": 2 }
            }
        },
        "pathing": {
            "rtree-threshold": 36,
            "nms-threshold": 0.7,
            "max-primary-distance": 400,
            "max-lateral-distance": 200,
            "spline-config": {
                "primary": { "tension": 0.5, "spacing": 50 },
                "lateral": { "tension": 0.5, "spacing": 20 }
            }
        }
    }
    """
    exit()
