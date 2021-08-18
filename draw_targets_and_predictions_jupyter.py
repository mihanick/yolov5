import numpy as np
from utils.plots import output_to_target,  plot_one_image

from tqdm import tqdm
from utils.datasets import create_dataloader
import torch
from models.experimental import attempt_load
from pathlib import Path
from utils.general import check_dataset, check_img_size, colorstr, increment_path, non_max_suppression
from utils.torch_utils import select_device, time_sync
import PIL
import numpy as np
from showArray import showarray

def PlotEv(weights='runs/train/exp/weights/best.pt'):
    with torch.no_grad():
        data='data/dwg.yaml'
        # model.pt path(s)
        #weights='yolov5s.pt'  # model.pt path(s)
        batch_size=16  # batch size
        imgsz=512  # inference size (pixels)
        conf_thres=0.5  # confidence threshold
        iou_thres=0.2  # NMS IoU threshold
        task='val'  # train, val, test, speed or study
        device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False  # treat as single-class dataset
        augment=False  # augmented inference
        save_txt=False  # save results to *.txt
        save_hybrid=False  # save label+prediction hybrid results to *.txt
        project='runs/val'  # save to project/name
        name='exp'  # save to project/name
        exist_ok=False  # existing project/name ok, do not increment
        dataloader=None
        save_dir=Path('')
        plots=True

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

        # Dataloader
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                        prefix=colorstr(f'{task}: '))[0]

        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
        
        s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.

        for batch_i, (img, targets, paths, _) in enumerate(tqdm(dataloader, desc=s)):
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

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_sync()
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)

            # Plot images

            f = save_dir / f'val_batch{batch_i}_pred_labels.jpg'  # predictions
            
            # 100% confidence for target by default (ones)
            tt1 = torch.ones(targets.shape[0], targets.shape[1]+1)
            # all other data from targets
            tt1[:,:-1] = targets

            tt2 =  output_to_target(out)
            trg =np.concatenate((tt1.cpu().numpy(), tt2))
            for img_no, img1 in enumerate(img):
                img1 = img1.cpu().numpy()
                #trg[num_targets x 7, where :,0 is image number in batch]
                plot_one_image(img1,trg[trg[:,0]==img_no], label = paths[img_no])
                # showarray(img1)
                
if __name__ == "__main__":
    PlotEv()