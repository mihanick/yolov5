git clone https://github.com/mihanick/yolov5
cd yolov5
pip install -r requirements.txt
pip install drawsvg
pip install gdown

gdown --id 1xMpsc2M8JgN84nh5xBAZj5gNvNgmNrCO
gdown --id 1zQNX6vJgnGT7h4TfeD__oGVzN8SMQNuA
python create_yolo_dataset_files.py

python train.py --img 512 --batch 24 --epochs 100 --data dwg.yaml --weights yolov5l.pt

python detect.py --weights runs/train/exp/weights/best.pt --source data/dwg/images/train, --imgsz=512, --conf_thres=0.1, --iou_thres=0.1, --exist_ok=True
