from __future__ import annotations

import numpy as np
import torch


def voc_ap(rec, prec, use_07_metric=True):
    """Average precision: VOC 2007 11-point or VOC 2010 area-under-curve."""
    rec = np.asarray(rec, dtype=np.float64)
    prec = np.asarray(prec, dtype=np.float64)
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0.0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
        return ap

    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    al, at, ar, ab = [a[:, i][:, None] for i in range(4)]
    bl, bt, br, bb = [b[:, i][None, :] for i in range(4)]
    inter_w = np.maximum(0.0, np.minimum(ar, br) - np.maximum(al, bl))
    inter_h = np.maximum(0.0, np.minimum(ab, bb) - np.maximum(at, bt))
    inter = inter_w * inter_h
    area_a = (ar - al) * (ab - at)
    area_b = (br - bl) * (bb - bt)
    return inter / (area_a + area_b - inter + 1e-16)


def average_precision_for_class(detections, ground_truths, iou_thresh=0.5, use_07_metric=True):
    """detections: list of (image_id, score, xyxy)
    ground_truths: dict image_id -> (M, 4) xyxy array (already filtered difficult)
    """
    npos = sum(v.shape[0] for v in ground_truths.values())
    if npos == 0:
        return 0.0, 0.0, 0.0

    detections = sorted(detections, key=lambda x: x[1], reverse=True)
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    seen = {k: np.zeros(v.shape[0], dtype=bool) for k, v in ground_truths.items()}

    for i, (image_id, _score, box) in enumerate(detections):
        gt = ground_truths.get(image_id)
        if gt is None or gt.shape[0] == 0:
            fp[i] = 1
            continue
        ious = _iou_matrix(np.asarray(box)[None, :], gt)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thresh and not seen[image_id][j]:
            tp[i] = 1
            seen[image_id][j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / npos
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)
    ap = voc_ap(rec, prec, use_07_metric=use_07_metric)
    return float(ap), float(rec[-1] if rec.size else 0.0), float(prec[-1] if prec.size else 0.0)


@torch.no_grad()
def evaluate_map(
    pred_by_image: dict[str, torch.Tensor],
    gt_by_image: dict[str, torch.Tensor],
    num_classes: int = 20,
    iou_thresh: float = 0.5,
    use_07_metric: bool = True,
):
    """pred tensor (D,6) xyxy-norm, score, class; gt tensor (M,6) xyxy-norm, class, difficult."""
    aps = []
    details = []
    for c in range(num_classes):
        dets = []
        gts = {}
        for image_id, pred in pred_by_image.items():
            if pred.numel():
                m = pred[:, 5].long() == c
                for row in pred[m].cpu():
                    dets.append((image_id, float(row[4]), row[:4].numpy()))
        for image_id, gt in gt_by_image.items():
            if gt.numel() == 0:
                gts[image_id] = np.zeros((0, 4))
                continue
            m = (gt[:, 4].long() == c) & (gt[:, 5] < 0.5)
            gts[image_id] = gt[m, :4].cpu().numpy() if m.any() else np.zeros((0, 4))
        ap, rec, prec = average_precision_for_class(dets, gts, iou_thresh, use_07_metric)
        aps.append(ap)
        details.append({"class": c, "ap": ap, "recall": rec, "precision": prec})
    return float(np.mean(aps) if aps else 0.0), aps, details
