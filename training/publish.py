import os
import shutil
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

def default_config():
	return {
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

def create_new_model(name, parent_json=None, use_parent_config = False):
    return {
        "name": name,
        "description": "",
        "uuid": str(uuid.uuid4()),
        "history": {
            "model": {
                "parent-model": parent_json["uuid"] if parent_json is not None else "",
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
        "configuration": parent_json["configuration"] if use_parent_config else default_config()
    }

def publish(args):
    print ("Publishing model \"{0}\"...".format(args.name))

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
        	print ("Loaded parent model") 
    else:
        parent_json = None

    if args.use_parent_config:
    	print ("Using parent json configuration")
    else:
    	print ("Using default configuration")

    # Create new model json description
    model_json = create_new_model(args.name, parent_json = parent_json, use_parent_config = args.use_parent_config)
    model_json["configuration"]["multi-plant"] = args.multi_plant

    # Calculate hash prefix for this model
    print ("Calculating hash prefix...", end="")
    digest = hashlib.sha256()
    with open(model_path, 'rb') as model_f:
        data = model_f.read(1024)
        while len(data) > 0:
            digest.update(data)
            data = model_f.read(1024)
    
    hash_prefix = digest.hexdigest()[:PREFIX_LENGTH]

    print (hash_prefix)

    # Model install directory
    model_file_name = "{0}-{1}.pth".format(args.name, hash_prefix)
    model_json["configuration"]["network"]["weights"] = model_file_name
    publish_model_path = publish_folder / model_json["configuration"]["network"]["weights"]

    # Copy model to output path
    print ("Publishing model to {0}".format(model_path))
    shutil.copyfile(model_path, publish_model_path)
    
    # Save json to output path
    print ("Saving JSON description to {0}".format(publish_json_path))
    with open(publish_json_path, 'w') as json_file:
    	json.dump(model_json, json_file, indent=4)

    print ("Publishing complete. To use this model place these files within the RootNav 2 inference/models folder.")