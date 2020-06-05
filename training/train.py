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
from rootnav2.loss import get_loss_function
from rootnav2.loader import get_loader 
from rootnav2.utils import get_logger
from rootnav2.metrics import runningScore, averageMeter
from rootnav2.augmentations import get_composed_augmentations
from rootnav2.schedulers import get_scheduler
from rootnav2.optimizers import get_optimizer

weights =[0.0021,0.1861,2.3898,0.6323,28.6333,31.0194]
class_weights = torch.FloatTensor(weights).cuda()
def show_example(img, gt_mask, pred_mask):
    img_np = img.cpu().data.numpy()

    img_np = np.transpose(img_np, [1,2,0])
    img_np = cv2.resize(img_np, (512,512))


    gt_mask_np = np.transpose(gt_mask, [1,2,0]) * 255.0   
    gt_mask_np = np.repeat(gt_mask_np, 3, 2)
    gt_mask_np = cv2.resize(gt_mask_np, (512,512))
    pred_mask = np.asarray(pred_mask) #.cpu().numpy()

    pred_mask = cv2.resize(pred_mask, (512,512))

    img = np.concatenate((img_np, gt_mask_np, pred_mask), axis=1)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    plt.savefig('3pic', bbox_inches="tight", pad_inches=0)

    
def train(cfg, logger, logdir):
    
    # Setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))
    np.random.seed(cfg.get('seed', 1337))
    random.seed(cfg.get('seed', 1337))

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup Augmentations
    augmentations = cfg['training'].get('augmentations', None)
    data_aug = get_composed_augmentations(augmentations)

    # Setup Dataloader
    data_loader = get_loader(cfg['data']['dataset'])
    data_path = cfg['data']['path']

    print ("Dataset Loading from...", data_path)

    t_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['train_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),
        augmentations=data_aug)

    v_loader = data_loader(
        data_path,
        is_transform=True,
        split=cfg['data']['val_split'],
        img_size=(cfg['data']['img_rows'], cfg['data']['img_cols']),)

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
    model.cuda()
    # Setup optimizer, lr_scheduler and loss function
    optimizer_cls = get_optimizer(cfg)
    optimizer_params = {k:v for k, v in cfg['training']['optimizer'].items() 
                        if k != 'name'}

    optimizer = optimizer_cls(model.parameters(), **optimizer_params)
    logger.info("Using optimizer {}".format(optimizer))

    scheduler = get_scheduler(optimizer, cfg['training']['lr_schedule'])

    loss_fn = get_loss_function(cfg)
    logger.info("Using loss {}".format(loss_fn))

    start_iter = 0
    if cfg['training']['resume'] is not None:
        if os.path.isfile(cfg['training']['resume']):
            logger.info(
                "Loading model and optimizer from checkpoint '{}'".format(cfg['training']['resume'])
            )
            checkpoint = torch.load(cfg['training']['resume'])
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_iter = checkpoint["epoch"]
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    cfg['training']['resume'], checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(cfg['training']['resume']))

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    best_iou = -100.0
    i = start_iter
    flag = True
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    while i <= cfg['training']['train_iters'] and flag:
        for (images, labels, hm, gt) in trainloader:
            i += 1
            start_ts = time.time()
            scheduler.step()
            model.train()
            images = images.to(device)
            labels = labels.to(device)
            hm = hm.to(device)
            gt = gt.to(device)

            outputs1, output2= model(images)
            

            optimizer.zero_grad()
            
            loss1 = loss(input=outputs1, target=labels)
                       
            loss2 = criterion(input=output2, target=hm)

            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
     

            optimizer.step()
 

            time_meter.update(time.time() - start_ts)


            if (i + 1) % cfg['training']['print_interval'] == 0:
                fmt_str = "Iter [{:d}/{:d}]  Loss: {:.4f}  Time/Image: {:.4f}"
                print_str = fmt_str.format(i + 1,
                                           cfg['training']['train_iters'], 
                                           loss1.item(),
                                           time_meter.avg / cfg['training']['batch_size'])

                print(print_str)
                logger.info(print_str)
                #writer.add_scalar('loss/train_loss', loss1.item(), i+1)
                time_meter.reset()

            if (i + 1) % cfg['training']['val_interval'] == 0 or \
               (i + 1) == cfg['training']['train_iters']:
                model.eval()
                with torch.no_grad():
                    for i_val, (images_val, labels_val, hm, gt) in tqdm(enumerate(valloader)):
                        images_val = images_val.to(device)
                        labels_val = labels_val.to(device)
                        gt = gt.to(device)
                        outputs1, outputs2= model(images_val)
                        

                        val_loss1 = loss(input=outputs1, target=labels_val)
                        

                        pred = outputs1.data.max(1)[1].cpu().numpy()
                        pred1 = np.squeeze(outputs1[0:1,:,:,:].data.max(1)[1].cpu().numpy(), axis=0)
                        gt = labels_val.data.cpu().numpy()


                        running_metrics_val.update(gt, pred)
                        val_loss_meter.update(val_loss1.item())


                #writer.add_scalar('loss/val_loss', val_loss_meter.avg, i+1)
                logger.info("Iter %d Loss: %.4f" % (i + 1, val_loss_meter.avg))

                score, class_iou = running_metrics_val.get_scores()
                for k, v in score.items():
                    print(k, v)
                    logger.info('{}: {}'.format(k, v))
                    #.add_scalar('val_metrics/{}'.format(k), v, i+1)

                for k, v in class_iou.items():
                    logger.info('{}: {}'.format(k, v))
                    #writer.add_scalar('val_metrics/cls_{}'.format(k), v, i+1)

                val_loss_meter.reset()
                running_metrics_val.reset()
                #####################picture ##################              
                decoded = v_loader.decode_segmap(pred1)              
                #############################################  
                out_path = 'snapshot.jpg'
                misc.imsave(out_path, decoded)
                #############################################

                if score["Mean IoU : \t"] >= best_iou:
                    best_iou = score["Mean IoU : \t"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/rootnav2.yml",
        help="Configuration file to use"
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    run_id = random.randint(1,100000)
    logdir = os.path.join('runs', os.path.basename(args.config)[:-4] , str(run_id))



    print('RUNDIR: {}'.format(logdir))

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Starting training')

    train(cfg, logger, logdir)
