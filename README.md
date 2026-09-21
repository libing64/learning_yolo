# YOLOv1 (PyTorch)

PyTorch re-implementation of YOLOv1 for PASCAL VOC 2007 + 2012, following [You Only Look Once](https://arxiv.org/abs/1506.02640) and the training/inference pipeline in [arXiv:2305.17786](https://arxiv.org/pdf/2305.17786v1).

## Setup

```bash
conda activate yolo
cd /home/libing/source/ml/yolo/learning_yolo
```

VOC is downloaded to `~/dataset/VOC`:

```bash
python -m yolo_v1.voc_download --root ~/dataset/VOC
```

## Train

```bash
# Full YOLOv1 (24 conv + 2 FC)
python train.py --data-root ~/dataset/VOC --model yolov1 --epochs 135 --batch-size 16

# Tiny / Fast YOLO (9 conv), as in the re-implementation paper
python train.py --data-root ~/dataset/VOC --model tiny --epochs 200 --batch-size 32 --output runs/yolov1-tiny
```

Training uses 2007+2012 `trainval`, validates mAP on 2007 `test`, SGD + OneCycle, mixed precision, and the original YOLO multi-part loss.

## Evaluate / detect

```bash
python evaluate.py --ckpt runs/yolov1/best.pt --data-root ~/dataset/VOC
python detect.py --ckpt runs/yolov1/best.pt --source /path/to/image.jpg --out outputs/detect
```
