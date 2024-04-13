import argparse, os
import torch
from models import ModelLoader
from run_rootnav import run_rootnav, list_action, info_action
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

if __name__ == '__main__':
    # Parser Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', action=list_action, nargs=0, help='List available models and exit')
    parser.add_argument('-i', '--info', action=info_action, nargs=1, help='Print detail on a single model and exit')
    parser.add_argument('--model', default="wheat_bluepaper", metavar='M', help="The trained model to use (default wheat_bluepaper)")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--segmentation_images', action='store_true', default=False, help='Reduce output files to minimum')
    parser.add_argument('--debug', action='store_true', default=False, help='Show additional debug messages')
    parser.add_argument('input_dir', type=str, help='Input directory', nargs="?")
    parser.add_argument('output_dir', type=str, help='Output directory', nargs="?")

    args = parser.parse_args()

    # Title
    logger.info("RootNav 2.1")

    # Input directory is required
    if not args.input_dir:
        logger.error("No input folder specified")
        parser.print_help()
        exit()

    if (args.debug):
        logger.setLevel(logging.DEBUG)
        logger.debug("Running in debug mode")

    output_dir = ''
    if not args.output_dir:
        logger.info("No output folder specified, will try and write output to " + args.input_dir + "_output")
        output_dir = args.input_dir + '_output'
    else:
        output_dir = args.output_dir

    if os.path.exists(output_dir):
        if os.listdir(output_dir):
            logger.warning(f"{output_dir} already exits and isn't empty, files may be overwritten")
        else:
            logger.debug(f"{output_dir} already exits and is empty")
    else:
        logger.debug(f"Creating output directory {output_dir}")
        os.makedirs(output_dir)

    # Check cuda configuration and notify if cuda is unavailable but they are trying to use it
    if not torch.cuda.is_available() and not args.no_cuda:
        logger.info("Cuda is not available, switching to CPU")
    use_cuda = torch.cuda.is_available() and not args.no_cuda

    # Load the model
    try:
        model_data = ModelLoader.get_model(args.model, gpu=use_cuda)
    except Exception as ex:
        logger.error(ex)
        exit()

    # Process
    run_rootnav(model_data, use_cuda, args, args.input_dir, output_dir)
