#!/usr/bin/env python3
"""Evaluate a YOLOv3 checkpoint on COCO val2017 (official COCOeval)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_v3.boxes import decode_outputs, non_max_suppression
from yolo_v3.config import YoloV3Config
from yolo_v3.dataset import CocoDetectionYolo, collate_fn
from yolo_v3.metrics import contiguous_to_coco_id, evaluate_coco, predictions_to_coco
from yolo_v3.model import build_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default=str(Path.home() / "dataset" / "coco2017"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--nms", type=float, default=0.45)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = YoloV3Config(
        **{k: v for k, v in ckpt.get("cfg", {}).items() if k in YoloV3Config.__dataclass_fields__}
    )
    cfg.data_root = args.data_root
    cfg.image_size = args.image_size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_set = CocoDetectionYolo(cfg.data_root, split="val", image_size=cfg.image_size, augment=False)
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    ann_file = Path(cfg.data_root) / "annotations" / "instances_val2017.json"
    id_map = contiguous_to_coco_id(ann_file)

    preds = []
    with torch.no_grad():
        for images, _targets, infos in tqdm(loader, desc="val"):
            images = images.to(device)
            outs = model(images)
            decoded = decode_outputs(outs, model.anchors, model.strides)
            dets = non_max_suppression(decoded, args.conf, args.nms, cfg.max_det)
            preds.extend(predictions_to_coco(dets, infos, images.shape[-1]))

    stats = evaluate_coco(preds, ann_file, id_map=id_map)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
