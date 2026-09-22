#!/usr/bin/env python3
"""Train YOLOv3 (pure PyTorch) on COCO 2017 to reproduce the paper setting."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_v3.boxes import decode_outputs, non_max_suppression
from yolo_v3.config import YoloV3Config
from yolo_v3.darknet_weights import load_pretrained_backbone
from yolo_v3.dataset import CocoDetectionYolo, collate_fn
from yolo_v3.loss import YOLOv3Loss
from yolo_v3.metrics import contiguous_to_coco_id, evaluate_coco, predictions_to_coco
from yolo_v3.model import build_model, count_parameters


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_ckpt(path: Path, model, optimizer, scaler, scheduler, epoch, best_ap, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_ap": best_ap,
            "cfg": vars(cfg),
        },
        tmp,
    )
    tmp.replace(path)


@torch.no_grad()
def run_eval(model, loader, device, cfg: YoloV3Config, ann_file: Path, id_map: dict):
    model.eval()
    preds = []
    anchors = model.anchors
    strides = model.strides
    for images, _targets, infos in loader:
        images = images.to(device, non_blocking=True)
        outs = model(images)
        decoded = decode_outputs(outs, anchors, strides)
        dets = non_max_suppression(
            decoded,
            conf_thresh=cfg.conf_thresh,
            iou_thresh=cfg.nms_thresh,
            max_det=cfg.max_det,
        )
        preds.extend(predictions_to_coco(dets, infos, images.shape[-1]))
    return evaluate_coco(preds, ann_file, id_map=id_map)


def parse_args():
    p = argparse.ArgumentParser(description="Train pure-PyTorch YOLOv3 on COCO")
    p.add_argument("--data-root", default=str(Path.home() / "dataset" / "coco2017"))
    p.add_argument("--model", default="yolov3", choices=["yolov3", "tiny"])
    p.add_argument("--epochs", type=int, default=273)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=416)
    p.add_argument("--no-multi-scale", action="store_true")
    p.add_argument("--output", default="runs/yolov3")
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--pretrained",
        default=None,
        help="Darknet-53 ImageNet weights: darknet53.conv.74 or converted .pt",
    )
    p.add_argument("--eval-interval", type=int, default=5)
    p.add_argument("--log-interval", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = YoloV3Config(
        data_root=args.data_root,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
        image_size=args.image_size,
        multi_scale=not args.no_multi_scale,
        output_dir=args.output,
        resume=args.resume,
        pretrained_backbone=args.pretrained,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,
    )
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(cfg), indent=2, default=str))

    train_set = CocoDetectionYolo(
        cfg.data_root,
        split="train",
        image_size=cfg.image_size,
        augment=True,
        multi_scale=cfg.multi_scale,
        multi_scale_range=cfg.multi_scale_range,
    )
    val_set = CocoDetectionYolo(
        cfg.data_root,
        split="val",
        image_size=cfg.image_size,
        augment=False,
        multi_scale=False,
    )
    if args.fast_dev_run:
        train_set.samples = train_set.samples[:64]
        val_set.samples = val_set.samples[:32]
        cfg.epochs = 1
        cfg.eval_interval = 1

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_model(cfg).to(device)
    if cfg.pretrained_backbone and not cfg.resume:
        load_pretrained_backbone(model, cfg.pretrained_backbone)
    print(
        f"Model={cfg.model} params={count_parameters(model):.1f}M device={device} "
        f"train={len(train_set)} val={len(val_set)}"
        + (f" pretrained={cfg.pretrained_backbone}" if cfg.pretrained_backbone else ""),
        flush=True,
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = YOLOv3Loss(
        anchors=model.anchors,
        strides=model.strides,
        num_classes=cfg.num_classes,
        ignore_thresh=cfg.ignore_thresh,
        lambda_coord=cfg.lambda_coord,
        lambda_obj=cfg.lambda_obj,
        lambda_noobj=cfg.lambda_noobj,
        lambda_cls=cfg.lambda_cls,
    )

    start_epoch = 1
    best_ap = -1.0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_ap = ckpt.get("best_ap", -1.0)
        print(f"Resumed from {cfg.resume} at epoch {start_epoch}", flush=True)

    ann_file = Path(cfg.data_root) / "annotations" / "instances_val2017.json"
    id_map = contiguous_to_coco_id(ann_file)
    log_path = out_dir / "log.jsonl"
    print(f"Logging to {log_path}", flush=True)

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        meters = {k: 0.0 for k in ("loss", "xy", "wh", "obj", "noobj", "cls")}
        seen = 0
        t0 = time.time()

        # multi-scale: random input size every 10 batches (paper / Darknet practice)
        sizes = list(range(cfg.multi_scale_range[0], cfg.multi_scale_range[1] + 1, 32))
        cur_size = cfg.image_size

        for step, (images, targets, _infos) in enumerate(train_loader):
            if cfg.multi_scale and step % 10 == 0:
                cur_size = random.choice(sizes)
            if images.shape[-1] != cur_size:
                images = F.interpolate(
                    images, size=(cur_size, cur_size), mode="bilinear", align_corners=False
                )

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            image_size = images.shape[-1]

            # warm-up LR
            if epoch <= cfg.warmup_epochs:
                warm = (epoch - 1 + (step + 1) / max(len(train_loader), 1)) / cfg.warmup_epochs
                for g in optimizer.param_groups:
                    g["lr"] = cfg.lr * warm

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                outs = model(images)
                parts = criterion(outs, targets, image_size=image_size)
                loss = parts["loss"]
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = images.size(0)
            seen += bs
            for k in meters:
                meters[k] += float(parts[k].item()) * bs

        if epoch > cfg.warmup_epochs:
            scheduler.step()

        train_stats = {k: v / max(seen, 1) for k, v in meters.items()}
        row = {
            "epoch": epoch,
            "train": train_stats,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }

        do_eval = epoch % cfg.eval_interval == 0 or epoch == cfg.epochs
        if do_eval:
            coco_stats = run_eval(model, val_loader, device, cfg, ann_file, id_map)
            row["coco"] = coco_stats
            ap = coco_stats["AP"]
            print(
                f"Epoch {epoch}/{cfg.epochs}: loss {train_stats['loss']:.3f}  "
                f"AP {ap:.4f} AP50 {coco_stats['AP50']:.4f}  "
                f"lr {row['lr']:.2e}  time {row['seconds']:.0f}s",
                flush=True,
            )
            if ap >= best_ap:
                best_ap = ap
                save_ckpt(out_dir / "best.pt", model, optimizer, scaler, scheduler, epoch, best_ap, cfg)
        elif epoch % max(1, args.log_interval) == 0:
            print(
                f"Epoch {epoch}/{cfg.epochs}: loss {train_stats['loss']:.3f} "
                f"(xy {train_stats['xy']:.3f} wh {train_stats['wh']:.3f} "
                f"obj {train_stats['obj']:.3f} cls {train_stats['cls']:.3f})  "
                f"lr {row['lr']:.2e}  time {row['seconds']:.0f}s",
                flush=True,
            )

        save_ckpt(out_dir / "last.pt", model, optimizer, scaler, scheduler, epoch, best_ap, cfg)
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"Done. best AP={best_ap:.4f}  checkpoints in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
