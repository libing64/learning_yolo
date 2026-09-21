from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

CLASS_TO_IDX = {name: i for i, name in enumerate(VOC_CLASSES)}


@dataclass
class YoloConfig:
    """Training / model hyper-parameters aligned with YOLOv1 + arXiv:2305.17786."""

    data_root: str = str(Path.home() / "dataset" / "VOC")
    model: str = "yolov1"  # yolov1 | tiny

    S: int = 7
    B: int = 2
    C: int = 20
    image_size: int = 448

    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5

    # From-scratch training uses BN (darknet yolov1.cfg) because ImageNet
    # pretraining from the original paper is skipped.
    batch_norm: bool = True
    fc_hidden: int | None = None  # default: 4096 (full) / 2048 (tiny)
    dropout: float = 0.5

    epochs: int = 135
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "onecycle"  # onecycle | cosine | original | none
    warmup_epochs: int = 5
    grad_clip: float = 5.0
    amp: bool = True

    conf_thresh: float = 0.1
    nms_thresh: float = 0.5
    eval_interval: int = 10

    seed: int = 42
    device: str = "cuda"
    output_dir: str = "runs/yolov1"
    resume: str | None = None

    extra_aug: bool = True
    jitter: float = 0.2

    class_names: tuple[str, ...] = field(default_factory=lambda: VOC_CLASSES)

    @property
    def out_dim(self) -> int:
        return self.S * self.S * (self.B * 5 + self.C)

    def resolved_fc_hidden(self) -> int:
        if self.fc_hidden is not None:
            return self.fc_hidden
        return 2048 if self.model == "tiny" else 4096
