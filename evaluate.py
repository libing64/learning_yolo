#!/usr/bin/env python3
"""Evaluate a YOLOv1 checkpoint on VOC 2007 test (VOC07 11-point mAP)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_v1.boxes import decode_predictions
from yolo_v1.config import VOC_CLASSES, YoloConfig
from yolo_v1.dataset import VOCYoloDataset, collate_train, collect_gt
from yolo_v1.metrics import evaluate_map
from yolo_v1.model import build_model
from yolo_v1.transforms import build_val_transforms


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-root", default=str(Path.home() / "dataset" / "VOC"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--nms", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = YoloConfig(**{k: v for k, v in ckpt.get("cfg", {}).items() if k in YoloConfig.__dataclass_fields__})
    cfg.data_root = args.data_root
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    val_set = VOCYoloDataset(
        cfg.data_root,
        years="2007",
        image_sets="test",
        transform=build_val_transforms(cfg.image_size),
        S=cfg.S,
        B=cfg.B,
        C=cfg.C,
        skip_difficult=False,
    )
    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_train,
    )
    gt_lookup = collect_gt(val_set)
    pred_by_image = {}
    with torch.no_grad():
        for images, _targets, infos in tqdm(loader, desc="eval"):
            images = images.to(device)
            pred = model(images)
            dets = decode_predictions(
                pred, S=cfg.S, B=cfg.B, C=cfg.C, conf_thresh=args.conf, nms_thresh=args.nms
            )
            for det, info in zip(dets, infos):
                pred_by_image[info["image_id"]] = det.cpu()

    mean_ap, aps, _ = evaluate_map(pred_by_image, gt_lookup, num_classes=cfg.C)
    per_class = {VOC_CLASSES[i]: round(aps[i], 4) for i in range(len(aps))}
    print(json.dumps({"mAP": round(mean_ap, 4), "AP": per_class}, indent=2))


if __name__ == "__main__":
    main()
