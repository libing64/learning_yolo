#!/usr/bin/env python3
"""Visualize train-time augmentations (image + boxes) for a VOC sample."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_v1.boxes import xywh_to_xyxy
from yolo_v1.config import VOC_CLASSES
from yolo_v1.dataset import VOCYoloDataset
from yolo_v1.transforms import build_train_transforms


def draw(image, boxes):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if boxes.numel():
        xyxy = xywh_to_xyxy(boxes[:, 1:5])
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1 * w, y1 * h, x2 * w, y2 * h], outline="lime", width=2)
            cls = int(boxes[i, 0].item())
            draw.text((x1 * w, max(0, y1 * h - 12)), VOC_CLASSES[cls], fill="lime")
    return img


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path.home() / "dataset" / "VOC"))
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", default="outputs/aug_preview.png")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    ds = VOCYoloDataset(args.data_root, years="2007", image_sets="trainval", encode=False, transform=None)
    tfm = build_train_transforms()
    idx = random.randrange(len(ds))
    image, boxes, info = ds[idx]
    cols = args.n + 1
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    axes[0].imshow(np.array(draw(image, boxes)))
    axes[0].set_title("original")
    axes[0].axis("off")
    for i in range(args.n):
        img_i, boxes_i = tfm(image.copy(), boxes.clone())
        # tfm ends with ToTensor; rebuild visualization from tensor
        if not isinstance(img_i, Image.Image):
            arr = (img_i.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            vis = Image.fromarray(arr)
        else:
            vis = img_i
        axes[i + 1].imshow(np.array(draw(vis, boxes_i)))
        axes[i + 1].set_title(f"aug {i + 1}")
        axes[i + 1].axis("off")
    fig.suptitle(info["image_id"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out} from {info['path']}")


if __name__ == "__main__":
    main()
