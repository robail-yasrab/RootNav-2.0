import os
from glob import glob
import json
import torch
from .hourglass import hg
from .utils import convert_state_dict

class ModelLoader():
    @staticmethod
    def list_models():
        # Scan model Directory
        model_dir = os.path.dirname(os.path.realpath(__file__))
        files = glob("{0}/*.json".format(model_dir))
        
        model_list = []
        for file in files:
            with open(file, 'r') as f:
                model_json = json.loads(f.read())
                model_list.append(model_json['name'])

        return model_list

    @staticmethod
    def get_model(name, gpu=True):
        model_dir = os.path.dirname(os.path.realpath(__file__))
        files = glob("{0}/*.json".format(model_dir))
        
        model_json = None

        model_list = []
        for file in files:
            with open(file, 'r') as f:
                current_json = json.loads(f.read())
                if current_json['name'] == name:
                    model_json = current_json
                    break

        supported_archs = ['hg']

        if model_json is None:
            raise (Exception("Model not found"))
        elif model_json['architecture'] not in supported_archs:
            raise (Exception("Model architecture {0} not supported".format(model_json['architecture'])))

        # Load model
        weights_file = "{0}/{1}".format(model_dir, model_json['weights'])
        print (weights_file)

        model = None
        if model_json['architecture'] == 'hg':
            model = hg()
            if gpu:
                state = convert_state_dict(torch.load(weights_file)['model_state'])
            else:
                state = convert_state_dict(torch.load(weights_file, map_location='cpu')['model_state'])
            model.load_state_dict(state)
            model.eval()
            model_json['model'] = model

        return model_json