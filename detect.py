#!/usr/bin/env python3
"""Run YOLOv1 detection on an image or a directory of images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_v1.boxes import decode_predictions
from yolo_v1.config import VOC_CLASSES, YoloConfig
from yolo_v1.model import build_model

PALETTE = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
    (0, 128, 255),
    (255, 0, 128),
    (128, 0, 255),
    (0, 255, 128),
    (64, 200, 80),
    (200, 64, 80),
    (80, 64, 200),
    (200, 200, 64),
    (64, 200, 200),
    (200, 64, 200),
    (32, 160, 240),
    (160, 32, 240),
]


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = YoloConfig(**{k: v for k, v in ckpt.get("cfg", {}).items() if k in YoloConfig.__dataclass_fields__})
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def draw_detections(bgr, dets, names=VOC_CLASSES):
    h, w = bgr.shape[:2]
    for row in dets:
        x1, y1, x2, y2, score, cls = row.tolist()
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        cid = int(cls)
        color = PALETTE[cid % len(PALETTE)]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cid]} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(bgr, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
        cv2.putText(bgr, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return bgr


@torch.no_grad()
def detect_image(model, cfg, image_path, device, conf, nms):
    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image.resize((cfg.image_size, cfg.image_size), Image.BILINEAR)).unsqueeze(0)
    pred = model(tensor.to(device))
    dets = decode_predictions(pred, cfg.S, cfg.B, cfg.C, conf, nms)[0].cpu()
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return draw_detections(bgr, dets)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source", required=True, help="image file or directory")
    p.add_argument("--out", default="outputs/detect")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise FileNotFoundError(f"No images under {src}")
    for path in paths:
        vis = detect_image(model, cfg, path, device, args.conf, args.nms)
        dest = out_dir / path.name
        cv2.imwrite(str(dest), vis)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
