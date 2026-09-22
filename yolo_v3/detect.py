#!/usr/bin/env python3
"""Run YOLOv3 detection on an image or directory (pure PyTorch)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_v3.boxes import decode_outputs, non_max_suppression
from yolo_v3.config import COCO_CLASSES, YoloV3Config
from yolo_v3.dataset import LetterBox
from yolo_v3.model import build_model


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = YoloV3Config(
        **{k: v for k, v in ckpt.get("cfg", {}).items() if k in YoloV3Config.__dataclass_fields__}
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def draw(bgr, dets, names=COCO_CLASSES):
    for row in dets:
        x1, y1, x2, y2, score, cls = row.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cid = int(cls)
        color = (0, 200, 0)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{names[cid] if cid < len(names) else cid} {score:.2f}"
        cv2.putText(bgr, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return bgr


@torch.no_grad()
def detect_one(model, cfg, path, device, conf, nms):
    image = Image.open(path).convert("RGB")
    lb = LetterBox(cfg.image_size)
    canvas, _, meta = lb(image, torch.zeros((0, 4)))
    tensor = TF.to_tensor(canvas).unsqueeze(0).to(device)
    outs = model(tensor)
    decoded = decode_outputs(outs, model.anchors, model.strides)
    dets = non_max_suppression(decoded, conf, nms, cfg.max_det)[0].cpu()
    # map back to original image
    pad_x, pad_y = meta["pad"]
    scale = meta["scale"]
    if dets.numel():
        dets = dets.clone()
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale
    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return draw(bgr, dets)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--out", default="outputs/detect_v3")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.ckpt, device)
    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = (
        [src]
        if src.is_file()
        else sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    )
    for path in paths:
        vis = detect_one(model, cfg, path, device, args.conf, args.nms)
        dest = out_dir / path.name
        cv2.imwrite(str(dest), vis)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
