#!/usr/bin/env python3
"""Plot training curves from a run directory's log.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_run(run_dir: Path) -> None:
    rows = load_rows(run_dir)
    epochs = [r["epoch"] for r in rows]
    train_loss = [r["train"]["loss"] for r in rows]
    train_coord = [r["train"]["coord"] for r in rows]
    train_obj = [r["train"]["obj"] for r in rows]
    train_cls = [r["train"]["cls"] for r in rows]
    train_noobj = [r["train"]["noobj"] for r in rows]
    lrs = [r["lr"] for r in rows]

    eval_rows = [r for r in rows if "mAP" in r]
    e_ep = [r["epoch"] for r in eval_rows]
    e_map = [r["mAP"] for r in eval_rows]
    e_val = [r["val"]["loss"] for r in eval_rows]
    best_i = max(range(len(e_map)), key=lambda i: e_map[i])
    best_ep, best_map = e_ep[best_i], e_map[best_i]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(epochs, train_loss, color="#1f77b4", lw=1.4, label="train loss")
    ax.plot(e_ep, e_val, color="#d62728", lw=1.6, marker="o", ms=3.5, label="val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / val loss")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(e_ep, e_map, color="#2ca02c", lw=1.8, marker="o", ms=3.5, label="VOC07 mAP")
    ax.axhline(0.65, color="#7f7f7f", ls="--", lw=1, label="target 0.65")
    ax.scatter([best_ep], [best_map], color="#d62728", zorder=5, s=40, label=f"best {best_map:.3f} @ {best_ep}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_ylim(0, 0.75)
    ax.set_title("VOC 2007 test mAP (11-point)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(run_dir / "curves_loss_map.png", dpi=140)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(epochs, train_coord, label="coord", lw=1.2)
    ax.plot(epochs, train_obj, label="obj", lw=1.2)
    ax.plot(epochs, train_noobj, label="noobj", lw=1.2)
    ax.plot(epochs, train_cls, label="cls", lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss component")
    ax.set_title("Training loss components")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    ax.plot(epochs, lrs, color="#9467bd", lw=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate (head)")
    ax.set_title("Learning rate schedule")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.savefig(run_dir / "curves_parts_lr.png", dpi=140)

    if eval_rows:
        aps = sorted(eval_rows[best_i]["AP"].items(), key=lambda x: x[1])
        names = [n for n, _ in aps]
        vals = [v for _, v in aps]
        fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
        colors = ["#d62728" if v < 0.4 else "#1f77b4" if v < 0.65 else "#2ca02c" for v in vals]
        ax.barh(names, vals, color=colors)
        ax.axvline(best_map, color="#7f7f7f", ls="--", lw=1, label=f"mAP {best_map:.3f}")
        ax.set_xlabel("AP")
        ax.set_xlim(0, 1.0)
        ax.set_title(f"Per-class AP (epoch {best_ep})")
        ax.grid(True, axis="x", alpha=0.3)
        ax.legend(frameon=False, loc="lower right")
        fig.savefig(run_dir / "curves_per_class_ap.png", dpi=140)

    print(f"best mAP={best_map:.4f} @ epoch {best_ep}")
    print(f"wrote curves under {run_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/yolov1_r50")
    args = parser.parse_args()
    plot_run(Path(args.run))


if __name__ == "__main__":
    main()
