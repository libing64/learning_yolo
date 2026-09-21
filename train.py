#!/usr/bin/env python3
"""Train YOLOv1 on PASCAL VOC 2007 trainval + 2012 trainval."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yolo_v1.boxes import decode_predictions
from yolo_v1.config import VOC_CLASSES, YoloConfig
from yolo_v1.dataset import VOCYoloDataset, collate_train, collect_gt
from yolo_v1.loss import YOLOv1Loss
from yolo_v1.metrics import evaluate_map
from yolo_v1.model import build_model
from yolo_v1.transforms import build_train_transforms, build_val_transforms
from yolo_v1.voc_download import download_voc, voc_ready


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_decay_params(model):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay


def build_scheduler(optimizer, cfg: YoloConfig, steps_per_epoch: int):
    name = cfg.scheduler.lower()
    if name == "none":
        return None
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if name == "original":
        # Original YOLO schedule, scaled down (no ImageNet pretrain).
        warmup = cfg.warmup_epochs
        milestones = [warmup + 75, warmup + 105]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    raise ValueError(f"Unknown scheduler {cfg.scheduler}")


@torch.no_grad()
def run_eval(model, loader, criterion, device, cfg: YoloConfig, gt_lookup):
    model.eval()
    totals = {k: 0.0 for k in ("loss", "coord", "obj", "noobj", "cls")}
    pred_by_image = {}
    n = 0
    for images, targets, infos in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        pred = model(images)
        parts = criterion(pred, targets)
        bs = images.size(0)
        n += bs
        for k in totals:
            totals[k] += float(parts[k].item()) * bs
        dets = decode_predictions(
            pred.detach(),
            S=cfg.S,
            B=cfg.B,
            C=cfg.C,
            conf_thresh=cfg.conf_thresh,
            nms_thresh=cfg.nms_thresh,
        )
        for det, info in zip(dets, infos):
            pred_by_image[info["image_id"]] = det.cpu()
    losses = {k: v / max(n, 1) for k, v in totals.items()}
    mean_ap, aps, _ = evaluate_map(pred_by_image, gt_lookup, num_classes=cfg.C)
    return losses, mean_ap, aps


def save_ckpt(path: Path, model, optimizer, scaler, scheduler, epoch, best_map, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "best_map": best_map,
        "cfg": vars(cfg),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv1 on PASCAL VOC 2007+2012")
    p.add_argument("--data-root", default=str(Path.home() / "dataset" / "VOC"))
    p.add_argument("--model", default="yolov1", choices=["yolov1", "tiny"])
    p.add_argument("--epochs", type=int, default=135)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--scheduler", default="onecycle")
    p.add_argument("--output", default="runs/yolov1")
    p.add_argument("--resume", default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-extra-aug", action="store_true")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--eval-interval", type=int, default=10)
    p.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Print epoch summary every N epochs (always print on eval).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fast-dev-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = YoloConfig(
        data_root=args.data_root,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.workers,
        image_size=args.image_size,
        scheduler=args.scheduler,
        output_dir=args.output,
        resume=args.resume,
        amp=not args.no_amp,
        extra_aug=not args.no_extra_aug,
        eval_interval=args.eval_interval,
        seed=args.seed,
        device=args.device,
    )
    log_interval = max(1, args.log_interval)
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(cfg), indent=2, default=str))

    if not voc_ready(Path(cfg.data_root).expanduser()):
        if args.no_download:
            raise FileNotFoundError(f"VOC not found at {cfg.data_root}")
        download_voc(cfg.data_root)

    train_set = VOCYoloDataset(
        cfg.data_root,
        years=("2007", "2012"),
        image_sets=("trainval", "trainval"),
        transform=build_train_transforms(cfg.image_size, cfg.extra_aug, cfg.jitter),
        S=cfg.S,
        B=cfg.B,
        C=cfg.C,
    )
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
        collate_fn=collate_train,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_train,
    )

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model={cfg.model} params={n_params:.1f}M device={device} train={len(train_set)} val={len(val_set)}")

    decay, no_decay = split_decay_params(model)
    optimizer = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        momentum=cfg.momentum,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))
    criterion = YOLOv1Loss(cfg.S, cfg.B, cfg.C, cfg.lambda_coord, cfg.lambda_noobj)

    start_epoch = 1
    best_map = -1.0
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", -1.0)
        if scheduler is not None and ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        elif scheduler is not None and cfg.scheduler == "onecycle":
            # Rebuild so LR schedule continues from the resumed epoch.
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=cfg.lr,
                epochs=cfg.epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.1,
                anneal_strategy="cos",
                last_epoch=ckpt["epoch"] * len(train_loader) - 1,
            )
        print(f"Resumed from {cfg.resume} at epoch {start_epoch}", flush=True)

    gt_lookup = collect_gt(val_set)
    log_path = out_dir / "log.jsonl"
    print(f"Logging to {log_path} (print every {log_interval} epochs)")

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        meters = {k: 0.0 for k in ("loss", "coord", "obj", "noobj", "cls")}
        seen = 0
        t0 = time.time()
        # No per-step tqdm: it floods tee'd train.log with \r progress spam.
        for images, targets, _ in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred = model(images)
                parts = criterion(pred, targets)
                loss = parts["loss"]
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if cfg.scheduler == "onecycle" and scheduler is not None:
                if prev_scale <= scaler.get_scale():
                    scheduler.step()

            bs = images.size(0)
            seen += bs
            for k in meters:
                meters[k] += float(parts[k].item()) * bs

        if cfg.scheduler in {"cosine", "original"} and scheduler is not None:
            scheduler.step()

        train_stats = {k: v / max(seen, 1) for k, v in meters.items()}
        row = {
            "epoch": epoch,
            "train": train_stats,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }

        do_eval = epoch % cfg.eval_interval == 0 or epoch == cfg.epochs
        do_print = do_eval or epoch % log_interval == 0 or epoch == start_epoch

        if do_eval:
            val_stats, mean_ap, aps = run_eval(model, val_loader, criterion, device, cfg, gt_lookup)
            row["val"] = val_stats
            row["mAP"] = mean_ap
            row["AP"] = {VOC_CLASSES[i]: aps[i] for i in range(len(aps))}
            if do_print:
                print(
                    f"Epoch {epoch}/{cfg.epochs}: train {train_stats['loss']:.3f}  "
                    f"val {val_stats['loss']:.3f}  mAP {mean_ap:.4f}  "
                    f"lr {row['lr']:.2e}  time {row['seconds']:.0f}s",
                    flush=True,
                )
            if mean_ap >= best_map:
                best_map = mean_ap
                save_ckpt(
                    out_dir / "best.pt", model, optimizer, scaler, scheduler, epoch, best_map, cfg
                )
        elif do_print:
            print(
                f"Epoch {epoch}/{cfg.epochs}: train {train_stats['loss']:.3f}  "
                f"lr {row['lr']:.2e}  time {row['seconds']:.0f}s",
                flush=True,
            )

        save_ckpt(out_dir / "last.pt", model, optimizer, scaler, scheduler, epoch, best_map, cfg)
        with log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"Done. best mAP={best_map:.4f}  checkpoints in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
