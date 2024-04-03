import sys, argparse, os
import torch
from models import ModelLoader
from run_rootnav import run_rootnav, list_action, info_action

if __name__ == '__main__':
    # Parser Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', action=list_action, nargs=0, help='List available models and exit')
    parser.add_argument('-i', '--info', action=info_action, nargs=1, help='Print detail on a single model and exit')
    parser.add_argument('--model', default="wheat_bluepaper", metavar='M', help="The trained model to use (default wheat_bluepaper)")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--segmentation_images', action='store_true', default=False, help='Reduce output files to minimum')
    parser.add_argument('input_dir', type=str, help='Input directory', nargs="?")
    parser.add_argument('output_dir', type=str, help='Output directory', nargs="?")

    args = parser.parse_args()

    # Title
    print("RootNav 2.1")
    sys.stdout.flush()

    # Input directory is required
    if not args.input_dir:
        parser.print_help()
        exit()

    output_dir = ''
    if not args.output_dir:
        print("No output folder specified, will try and write output to "+args.input_dir+"_output")
        output_dir = args.input_dir + '_output'
    else:
        output_dir = args.output_dir

    if os.path.exists(output_dir):
        if os.listdir(output_dir):
            print(output_dir+" already exits and isn't empty, please delete if you can or specify output folder name")
            exit()
        else:
            print(output_dir+" exits and is empty")
    else:
        print(output_dir+" does not exit, creating")
        os.makedirs(output_dir)

    # Check cuda configuration and notify if cuda is unavailable but they are trying to use it
    if not torch.cuda.is_available() and not args.no_cuda:
        print ("Cuda is not available, switching to CPU")
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    # Load the model
    try:
        model_data = ModelLoader.get_model(args.model, gpu=use_cuda)
    except Exception as ex:
        print (ex)
        exit()

    # Process
    run_rootnav(model_data, use_cuda, args.input_dir, output_dir, args.segmentation_images)
