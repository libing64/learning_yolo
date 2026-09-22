from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# COCO 80 class names (paper / Darknet order)
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

# Paper / Darknet COCO anchors, grouped by scale (P3/8, P4/16, P5/32)
ANCHORS_COCO = (
    ((10, 13), (16, 30), (33, 23)),
    ((30, 61), (62, 45), (59, 119)),
    ((116, 90), (156, 198), (373, 326)),
)

STRIDES = (8, 16, 32)


@dataclass
class YoloV3Config:
    """Hyper-parameters aligned with YOLOv3 paper + common Darknet/Ultralytics practice."""

    data_root: str = str(Path.home() / "dataset" / "coco2017")
    model: str = "yolov3"  # yolov3 | tiny

    num_classes: int = 80
    image_size: int = 416  # paper also reports 320 / 608; multi-scale uses [320..608]
    multi_scale: bool = True
    multi_scale_range: tuple[int, int] = (320, 608)

    anchors: tuple = ANCHORS_COCO
    strides: tuple[int, ...] = STRIDES

    # Loss weights (Darknet-style)
    lambda_coord: float = 1.0
    lambda_obj: float = 1.0
    lambda_noobj: float = 1.0
    lambda_cls: float = 1.0
    ignore_thresh: float = 0.5

    epochs: int = 273  # MMDetection / common COCO schedule for YOLOv3
    batch_size: int = 8
    num_workers: int = 8
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 3
    scheduler: str = "cosine"  # cosine | step
    grad_clip: float = 10.0
    amp: bool = True

    conf_thresh: float = 0.001
    nms_thresh: float = 0.45
    max_det: int = 300
    eval_interval: int = 5

    seed: int = 42
    device: str = "cuda"
    output_dir: str = "runs/yolov3"
    resume: str | None = None
    pretrained_backbone: str | None = None  # optional Darknet-53 .pt path

    class_names: tuple[str, ...] = field(default_factory=lambda: COCO_CLASSES)

    @property
    def num_anchors_per_scale(self) -> int:
        return len(self.anchors[0])
