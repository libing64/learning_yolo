# YOLOv3 (pure PyTorch)

Educational **YOLOv3** implementation in pure PyTorch — **no Ultralytics model code**.
Architecture follows [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
and the Darknet / [ultralytics/yolov3](https://github.com/ultralytics/yolov3) layer layout
(as a reference only).

Paper COCO result (608×608, Darknet-53): **AP 33.0 / AP50 57.9**.

## Layout

```
yolo_v3/
  model.py      # Darknet-53 + 3-scale head (and YOLOv3-tiny)
  darknet_weights.py  # load darknet53.conv.74 ImageNet backbone
  loss.py       # multi-scale BCE / MSE loss
  dataset.py    # COCO 2017 letterbox loader + multi-scale
  boxes.py      # decode + NMS
  metrics.py    # official pycocotools COCOeval
  train.py      # COCO train loop
  val.py        # COCO val2017 eval
  detect.py     # image inference
  config.py     # anchors, hyper-parameters
```

## Data

Uses local COCO 2017:

```
~/dataset/coco2017/
  train2017/  val2017/  annotations/instances_*.json
```

## Pretrained Darknet-53

Paper training starts from ImageNet-pretrained Darknet-53 (`darknet53.conv.74`):

```bash
# Already downloaded under weights/; re-convert if needed:
python -m yolo_v3.darknet_weights \
  --weights weights/darknet53.conv.74 \
  --out weights/darknet53.pt
```

## Train (paper-style)

```bash
conda activate yolo
cd /home/libing/source/ml/yolo/learning_yolo

# Full YOLOv3 on COCO (ImageNet backbone + multi-scale 320–608, cosine 273 epochs)
python -m yolo_v3.train \
  --data-root ~/dataset/coco2017 \
  --model yolov3 \
  --pretrained weights/darknet53.conv.74 \
  --epochs 273 \
  --batch-size 8 \
  --image-size 416 \
  --lr 1e-3 \
  --output runs/yolov3

# Tiny
python -m yolo_v3.train --model tiny --batch-size 32 --output runs/yolov3-tiny
```

Training matches the paper recipe at a high level:

- **ImageNet-pretrained Darknet-53** backbone (`--pretrained`)
- full images (letterbox), **no hard-negative mining**
- **multi-scale** training (320…608, step 32)
- color / flip augmentation + batch norm
- SGD + cosine LR, COCO train2017 → val2017 COCOeval

## Eval / detect

```bash
python -m yolo_v3.val --ckpt runs/yolov3/best.pt --data-root ~/dataset/coco2017 --image-size 608
python -m yolo_v3.detect --ckpt runs/yolov3/best.pt --source path/to.jpg --out outputs/detect_v3
```

## Notes

- Reproducing the paper’s **33.0 AP** typically needs long COCO training (hundreds of
  epochs), multi-GPU or large batches, and preferably ImageNet-pretrained Darknet-53.
  This codebase trains from scratch on one GPU by default; expect lower AP early on.
- Eval reports COCO **AP / AP50 / AP75 / APs / APm / APl** via `pycocotools`.
