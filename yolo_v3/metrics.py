from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


@torch.no_grad()
def predictions_to_coco(
    dets: list[torch.Tensor],
    infos: list[dict],
    input_size: int,
) -> list[dict]:
    """Convert network-input-pixel detections back to original image xywh."""
    results = []
    for det, info in zip(dets, infos):
        if det.numel() == 0:
            continue
        pad_x, pad_y = info["pad"]
        scale = info["scale"]
        image_id = int(info["image_id"])
        boxes = det[:, :4].clone()
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        w0, h0 = info["orig_size"]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w0)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h0)
        for row, box in zip(det, boxes):
            x1, y1, x2, y2 = box.tolist()
            score = float(row[4])
            cls = int(row[5])
            results.append(
                {
                    "image_id": image_id,
                    "category_id": cls,  # remapped later to COCO ids
                    "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                    "score": score,
                }
            )
    return results


def contiguous_to_coco_id(ann_file: str | Path) -> dict[int, int]:
    coco = COCO(str(ann_file))
    cats = sorted(coco.loadCats(coco.getCatIds()), key=lambda c: c["id"])
    return {i: c["id"] for i, c in enumerate(cats)}


def evaluate_coco(
    pred_json_rows: list[dict],
    ann_file: str | Path,
    id_map: dict[int, int] | None = None,
) -> dict[str, float]:
    """Run official COCOeval. pred category_id should already be COCO ids, or pass id_map."""
    if not pred_json_rows:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0}

    rows = []
    for r in pred_json_rows:
        item = dict(r)
        if id_map is not None:
            item["category_id"] = id_map[int(item["category_id"])]
        rows.append(item)

    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(rows)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    stats = ev.stats  # AP, AP50, AP75, APs, APm, APl, ...
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    }
