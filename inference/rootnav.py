import sys, os
import torch
import argparse
from models import ModelLoader
from run_rootnav import run_rootnav, list_action, info_action

if __name__ == '__main__':
    print("RootNav 2.0")
    sys.stdout.flush()

    # Parser Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', action=list_action, nargs=0, help='List available models and exit')
    parser.add_argument('-i', '--info', action=info_action, nargs=1, help='Print detail on a single model')
    parser.add_argument('--model', default="wheat_bluepaper", metavar='M', help="The trained model to use (default wheat_bluepaper)")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--no_crf', action='store_true', default=False, help='disables CRF post-processing')
    parser.add_argument('input_dir', type=str, help='Input directory', nargs="?")
    parser.add_argument('output_dir', type=str, help='Output directory', nargs="?")
    parser.add_argument('--no_segmentation_images', action='store_true', default=False, help='Reduce output files to minimum')

    args = parser.parse_args()

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
