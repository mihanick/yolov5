from threading import Thread

import numpy as np
from utils.plots import output_to_target, plot_images
from val import process_batch
from tqdm import tqdm
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.datasets import create_dataloader
import torch
from models.experimental import attempt_load
from pathlib import Path
from utils.general import check_dataset, check_img_size, colorstr, increment_path, non_max_suppression, scale_coords, xywh2xyxy
from utils.torch_utils import select_device, time_sync
from utils.loggers import Loggers


with torch.no_grad():
    data='data/dwg.yaml'
    weights='runs/train/exp/weights/best.pt'  # model.pt path(s)
    #weights='yolov5s.pt'  # model.pt path(s)
    batch_size=16  # batch size
    imgsz=512  # inference size (pixels)
    conf_thres=0.1  # confidence threshold
    iou_thres=0.6  # NMS IoU threshold
    task='val'  # train, val, test, speed or study
    device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    single_cls=False  # treat as single-class dataset
    augment=False  # augmented inference
    verbose=False  # verbose output
    save_txt=False  # save results to *.txt
    save_hybrid=False  # save label+prediction hybrid results to *.txt
    save_conf=False  # save confidences in --save-txt labels
    save_json=False  # save a COCO-JSON results file
    project='runs/val'  # save to project/name
    name='exp'  # save to project/name
    exist_ok=True  # existing project/name ok, do not increment
    dataloader=None
    save_dir=Path('')
    plots=True
    loggers=Loggers()
    compute_loss=None

    # Initialize/load model and set device
    device = select_device(device, batch_size=batch_size)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(imgsz, s=gs)  # check image size

    # Data
    data = check_dataset(data)  # check

    # Configure
    model.eval()

    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
    dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                    prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t_ = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        out, train_out = model(img, augment=augment)  # inference and training outputs
        t1 += time_sync() - t

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # Run NMS
        targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t2 += time_sync() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, iouv)
                if plots:
                    confusion_matrix.process_batch(predn, labelsn)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            loggers.on_val_batch_end(pred, predn, path, names, img[si])

        # Plot images
        if plots and batch_i < 3:
            #f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
            #Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
            f = save_dir / f'val_batch{batch_i}_pred_labels.jpg'  # predictions
            
            # 100% confidence for target by default (ones)
            tt1 = torch.ones(targets.shape[0], targets.shape[1]+1)
            # all other data from targets
            tt1[:,:-1] = targets

            tt2 =  output_to_target(out)
            trg =np.concatenate((tt1.cpu().numpy(), tt2))
            Thread(target=plot_images, args=(img, trg, paths, f, names, imgsz, batch_size, 1, conf_thres), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image

    shape = (batch_size, 3, imgsz, imgsz)
    print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        loggers.on_val_end()
