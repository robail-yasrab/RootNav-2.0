import yaml
import torch
from torch.utils import data
from rootnav2.hourglass import hg
from rootnav2.loader import get_loader 
from rootnav2.metrics import runningScore, averageMeter, LocalisationAccuracyMeter
from rootnav2.utils import convert_state_dict, dict_collate
from rootnav2.accuracy import nonmaximalsuppression as nms, rrtree, evaluate_points

def test(args):
    # Load Config
    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)

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
    print("Loading model weights")
    model = hg()
    state = convert_state_dict(torch.load(args.model)["model_state"])
    model.load_state_dict(state)
    
    model = torch.nn.DataParallel(model)
    model.to(device)

    model.eval()

    val_loss_meter = averageMeter()
    time_meter = averageMeter()

    print("Starting test")
    
    accuracy_meter = LocalisationAccuracyMeter(3) # Seed, Primary, Lateral
    total = 0

    with torch.no_grad():
        for i_val, (images, masks, annotations) in enumerate(testloader):
            images = images.to(device)
            outputs = model(images)[-1].data.cpu()
            total += outputs.size(0)

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
                    raw_points = nms(current[channel_idx], 0.7)
                    pred = rrtree(raw_points, 36)
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

        print ("\r\nProcessed {0} images".format(total))
        metrics, class_iou = running_metrics.get_scores()

        print ("\r\nSegmentation Results:")
        print ("Overall Accuracy: {0:.4f}".format(metrics['oacc']))
        print ("Mean Accuracy:    {0:.4f}".format(metrics['macc']))
        print ("FreqW Accuracy:   {0:.4f}".format(metrics['facc']))
        print ("Root Mean IoU:    {0:.4f}".format((class_iou[0] + class_iou[1] + class_iou[3]) / 3))

        print ("\r\nLocalisation Results:")
        format_string = "{0}\tPrecision: {1:.4f}  Recall: {2:.4f}  F1: {3:.4f}"

        for (channel, results) in zip(["Seeds", "Primary", "Lateral"], accuracy_meter.f1()):
            print (format_string.format(channel, *results))
        
        print("\r\nTest completed succesfully")
