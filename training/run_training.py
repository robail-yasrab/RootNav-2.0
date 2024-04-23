import os
import sys
import yaml
import time
import shutil
import torch
import random
import datetime
import numpy as np
from torch.utils import data
from rootnav2.hourglass import hg
from rootnav2.loss import get_loss_function
from rootnav2.loader import get_loader 
from rootnav2.utils import decode_segmap
from rootnav2.metrics import runningScore, averageMeter
from rootnav2.schedulers import get_scheduler
from rootnav2.optimizers import get_optimizer
from PIL import Image
from tensorboardX import SummaryWriter
import logging

# Class weights
weights = [0.0007,1.6246,0.7223,0.1789,1.748,12.9261] #[0.0021,0.1861,2.3898,0.6323,28.6333,31.0194]

def train(args):
    # Load Config
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)


    # Create log and output directory
    run_id = random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))
    
    # Tensorboard summaries
    writer = SummaryWriter(log_dir=logdir)

    # Logging 
    logger = logging.getLogger()

    # Initialise log file
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    file_handler = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler) 

    if (args.debug):
        logger.setLevel(logging.DEBUG)
        logger.debug("Running in debug mode")

    logger.info(f"Running in working directory {logdir}")

    if not os.path.exists(logdir):
        logger.debug(f"Creating directory {logdir}")
        os.makedirs(logdir)

    logger.debug(f"Copying config file to working directory")    
    shutil.copy(args.config, logdir)

    logger.info('Starting training')

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        logger.warning("No cuda available. Training will be very slow on a CPU.")
    
    # Class weights
    class_weights = torch.FloatTensor(weights).to(device)

    # Setup seeds
    torch.manual_seed(cfg.get('seed', 65537))
    torch.cuda.manual_seed(cfg.get('seed', 65537))
    np.random.seed(cfg.get('seed', 65537))
    random.seed(cfg.get('seed', 65537))

    # Is hflip in use?
    augmentations = cfg['training'].get('augmentations', 0.0)
    if augmentations is not None:
        hflip = augmentations.get('hflip', 0.0)
    else:
        hflip = 0.0

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    logger.info(f"Dataset Loading from {data_path}")

    t_loader = data_loader(
        data_path,
        split='train',
        hflip=hflip)

    v_loader = data_loader(
        data_path,
        split='valid')

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader,
                                  batch_size=cfg['training']['batch_size'], 
                                  num_workers=cfg['training']['n_workers'], 
                                  shuffle=True)

    valloader = data.DataLoader(v_loader, 
                                batch_size=cfg['training']['batch_size'], 
                                num_workers=cfg['training']['n_workers'])

    # Setup Metrics
    running_metrics_val = runningScore(n_classes)

    # Setup Model
    model =  hg()

    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.to(device)

    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.debug("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])
    logger.debug("Using scheduler {}".format(scheduler))

    loss_fn = get_loss_function(cfg)
    logger.debug("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            if (args.resume_iterations) and checkpoint["epoch"] < cfg['training']['train_iters']:
                start_iter = checkpoint["epoch"]
                logger.info(
                "Loaded checkpoint '{}' (current iteration {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                    )
                )
            elif (args.resume_iterations) and checkpoint["epoch"] >= cfg['training']['train_iters']:
                logger.error(f"Listed target iteration number of {cfg['training']['train_iters']} is lower than the iteration number of the transfer learning model {checkpoint['epoch']}. Please rerun training without continuing iterations or by specifying a higher total number of iterations")
                exit()
        else:
            logger.warning("No checkpoint found at '{}'".format(cfg['training']['resume']))
            logger.info("plant models can be downloaded using the following links:")
            print("https://cvl.cs.nott.ac.uk/resources/trainedmodels/arabidopsis_plate-ea874d94.pth")
            print("https://cvl.cs.nott.ac.uk/resources/trainedmodels/osr_bluepaper-083ed788.pth")
            print("https://cvl.cs.nott.ac.uk/resources/trainedmodels/wheat_bluepaper-6d109612.pth")
            exit()

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    bce_criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    mse_criterion = torch.nn.MSELoss(reduction='mean').to(device)

    logger.info("Starting training")
    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels, hm) in trainloader:
            i += 1
            start_ts = time.time()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            hm = hm.to(device)

            outputs= model(images)
            out_main= outputs[-1]
            sys.stdout.flush()
            
            optimizer.zero_grad()
            
            loss1 = bce_criterion(input=out_main, target=labels)

            out5= out_main[:,5:6,:,:] 
            out4= out_main[:,4:5,:,:]
            out2= out_main[:,2:3,:,:] 

            tips = torch.cat((out2, out4,  out5), 1)
            loss2 = mse_criterion(input=tips, target=hm)

            loss1.backward(retain_graph=True)
            loss2.backward()

            optimizer.step()
            scheduler.step()

            time_meter.update(time.time() - start_ts)

            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'], 
                                           loss1.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                logger.info(print_str)
                writer.add_scalar('loss/train_loss', loss1.item(), i+1)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                model.eval()
                logger.info("Validation:")
                with torch.no_grad():
                    for images_val, labels_val, hm in valloader:
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        
                        outputs = model(images_val)
                        outputs1= outputs[-1]
                        
                        val_loss1 = bce_criterion(input=outputs1, target=labels_val)
                        
                        pred = outputs1.data.max(1)[1].cpu().numpy()
                        pred1 = np.squeeze(outputs1[0:1,:,:,:].data.max(1)[1].cpu().numpy(), axis=0)
                        gt = labels_val.data.cpu().numpy()

                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss1.item())


                writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                logger.info(f"Iter [{i+1}] Loss: {val_loss_meter.avg:.4f}")

                score, class_iou = running_metrics_val.get_scores()

                results = ["Overall Accuracy: {0:.6f}".format(score['oacc']),
                           "Mean Accuracy:    {0:.6f}".format(score['macc']),
                           "FreqW Accuracy:   {0:.6f}".format(score['facc']),
                           "Mean IoU:         {0:.6f}".format(score['miou'])]

                for log_entry in results:
                    logger.info(log_entry)

                logger.debug("Class IoU scores:")
                for k, v in class_iou.items():
                    logger.debug('{}: {}'.format(k, v))
                    writer.add_scalar('val_metrics/cls_{}'.format(k), v, i+1)

                val_loss_meter.reset()
                running_metrics_val.reset()

                if (args.output_example):
                    # Output example image
                    channel_bindings = {'segmentation': {'Background': 0, 'Primary': 3, 'Lateral': 1}, 'heatmap': {'Seed': 5, 'Primary': 4, 'Lateral': 2}}
                    decoded = decode_segmap(np.array(pred1, dtype=np.uint8), channel_bindings)
                    decoded = Image.fromarray(decoded, 'RGBA')
                    example_path = os.path.join(logdir, 'validation_example.png')
                    decoded.save(example_path)
                    logger.info("Example image saved")

                if score['miou'] >= best_iou:
                    best_iou = score['miou']
                    state = {
                        "epoch": i + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_iou": best_iou,
                    }
                    save_path = os.path.join(writer.file_writer.get_logdir(),
                                             "{}_{}_best_model.pkl".format(
                                                 cfg['model']['arch'],
                                                 cfg['data']['dataset']))
                    torch.save(state, save_path)

            if (i + 1) == cfg['training']['train_iters']:
                flag = False
                break

    file_handler.close()
