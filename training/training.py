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
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils import data
from tqdm import tqdm
from rootnav2.hourglass import hg
import cv2
from rootnav2.loss import get_loss_function
from rootnav2.loader import get_loader 
from rootnav2.utils import decode_segmap
from rootnav2.metrics import runningScore, averageMeter
from rootnav2.schedulers import get_scheduler
from rootnav2.optimizers import get_optimizer
from pathlib import Path
from publish import publish
from test import test
from run_training import train
from PIL import Image
import logging

class LogFormatter(logging.Formatter):
    def format(self, record):
        self.datefmt='%H:%M:%S'
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            color = {
                logging.WARNING: 33,
                logging.ERROR: 31,
                logging.FATAL: 31,
                logging.DEBUG: 36
            }.get(record.levelno, 0)
            self._style._fmt = f"[%(asctime)s.%(msecs)03d] \033[{color}m%(levelname)s\033[0m: %(message)s"
        return super().format(record)

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(LogFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RootNav 2 Training")
    subparsers = parser.add_subparsers(title="Mode")

    # Train sub command
    parser_train = subparsers.add_parser('train', help='Train new models')
    parser_train.add_argument("--config", nargs="?", type=str, default="configs/rootnav2.yml", help="Configuration file to use")
    parser_train.add_argument('--output-example', action='store_true', help="Whether or not to output an example image each validation step")
    parser_train.add_argument('--debug', action='store_true', default=False, help='Show additional debug messages')
    parser_train.add_argument("--resume-iterations", action='store_true', default=False, help='Resume with the iteration count from the model selected with transfer learning')
    parser_train.set_defaults(func=train)

    # Publish sub command
    parser_publish = subparsers.add_parser('publish', help='Publish already trained models')
    parser_publish.add_argument('--name', default="published_model", metavar='N', help="The name of the new published model")
    parser_publish.add_argument('--parent', default=None, metavar='P', help="The name of the parent model used to begin training")
    parser_publish.add_argument('--model', metavar='M', help="The trained weights file to publish")
    parser_publish.add_argument('--multi-plant', action='store_true', help="Whether or not images are expected to contain multiple plants")
    parser_publish.add_argument('--use-parent-config', action='store_true', help="Whether or not to use the parent pathing and network configuration, or to use default values")
    parser_publish.add_argument('output_dir', default='./', type=str, help='Output directory')
    parser_publish.set_defaults(func=publish)

    # Testing sub command
    parser_test = subparsers.add_parser('test', help='Test already trained models')
    parser_test.add_argument('--model', metavar='M', help="The trained weights file to test")
    parser_test.add_argument("--config", nargs="?", type=str, default="configs/rootnav2.yml", help="Configuration file to use")
    parser_test.set_defaults(func=test)

    args = parser.parse_args()
    args.func(args)
