from __future__ import annotations

import torch
from torchvision.ops import nms


def encode_target(
    boxes: torch.Tensor,
    S: int = 7,
    B: int = 2,
    C: int = 20,
) -> torch.Tensor:
    """Encode YOLO-format boxes (N, 5) = class, cx, cy, w, h into an SxSx(B*5+C) tensor.

    Each occupied cell stores one ground-truth box in the first predictor slot:
    [x_cell, y_cell, w, h, 1,  0,0,0,0,0,  class_one_hot].
    x,y are relative to the cell; w,h are relative to the whole image.
    """
    target = torch.zeros(S, S, B * 5 + C, dtype=torch.float32)
    if boxes.numel() == 0:
        return target

    for box in boxes:
        cls = int(box[0].item())
        cx, cy, w, h = box[1:].tolist()
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            continue
        j = min(S - 1, int(cx * S))
        i = min(S - 1, int(cy * S))
        if target[i, j, 4] == 1:
            continue
        target[i, j, 0] = cx * S - j
        target[i, j, 1] = cy * S - i
        target[i, j, 2] = w
        target[i, j, 3] = h
        target[i, j, 4] = 1.0
        if 0 <= cls < C:
            target[i, j, B * 5 + cls] = 1.0
    return target


def cell_to_image_xywh(boxes: torch.Tensor, S: int) -> torch.Tensor:
    """Convert cell-relative (x, y, w, h) on an SxS grid to image-normalized xywh.

    boxes: (N, S, S, B, 4) or (N, S, S, 4)
    """
    squeeze_b = boxes.dim() == 4
    if squeeze_b:
        boxes = boxes.unsqueeze(-2)
    *prefix, four = boxes.shape
    assert four == 4
    n, s1, s2, b, _ = boxes.shape
    device = boxes.device
    gy, gx = torch.meshgrid(
        torch.arange(s1, device=device, dtype=boxes.dtype),
        torch.arange(s2, device=device, dtype=boxes.dtype),
        indexing="ij",
    )
    gx = gx.view(1, s1, s2, 1)
    gy = gy.view(1, s1, s2, 1)
    cx = (boxes[..., 0] + gx) / S
    cy = (boxes[..., 1] + gy) / S
    w = boxes[..., 2]
    h = boxes[..., 3]
    out = torch.stack([cx, cy, w, h], dim=-1)
    return out.squeeze(-2) if squeeze_b else out


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack(
        [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        dim=-1,
    )


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)],
        dim=-1,
    )


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between xyxy boxes. Broadcasts on the last dimension.

    a: (..., 4), b: (..., 4) -> (...,)
    """
    al, at, ar, ab = a.unbind(-1)
    bl, bt, br, bb = b.unbind(-1)
    inter_w = (torch.min(ar, br) - torch.max(al, bl)).clamp(min=0)
    inter_h = (torch.min(ab, bb) - torch.max(at, bt)).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (ar - al).clamp(min=0) * (ab - at).clamp(min=0)
    area_b = (br - bl).clamp(min=0) * (bb - bt).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-6)


def clip_xywh_normalized(boxes: torch.Tensor) -> torch.Tensor:
    """Clip class,cx,cy,w,h boxes to the unit image and drop invalid ones."""
    if boxes.numel() == 0:
        return boxes
    xyxy = xywh_to_xyxy(boxes[:, 1:5]).clamp(0.0, 1.0)
    xywh = xyxy_to_xywh(xyxy)
    out = torch.cat([boxes[:, :1], xywh], dim=1)
    keep = (out[:, 3] > 1e-3) & (out[:, 4] > 1e-3)
    keep &= (out[:, 1] > 0.0) & (out[:, 1] < 1.0)
    keep &= (out[:, 2] > 0.0) & (out[:, 2] < 1.0)
    return out[keep]


def decode_predictions(
    pred: torch.Tensor,
    S: int = 7,
    B: int = 2,
    C: int = 20,
    conf_thresh: float = 0.1,
    nms_thresh: float = 0.5,
) -> list[torch.Tensor]:
    """Decode a batch of SxSx(B*5+C) tensors to per-image detections.

    Returns a list of (D, 6) tensors: x1, y1, x2, y2, score, class  (normalized xyxy).
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    n = pred.shape[0]
    pred = pred.reshape(n, S, S, B * 5 + C)
    box_raw = pred[..., : B * 5].reshape(n, S, S, B, 5)
    cls_prob = pred[..., B * 5 :].clamp(min=0)  # (N,S,S,C)
    cls_score, cls_idx = cls_prob.max(dim=-1)  # (N,S,S)

    xywh = cell_to_image_xywh(box_raw[..., :4], S)
    xyxy = xywh_to_xyxy(xywh).clamp(0.0, 1.0)
    conf = box_raw[..., 4]
    scores = conf * cls_score.unsqueeze(-1)  # (N,S,S,B)
    classes = cls_idx.unsqueeze(-1).expand_as(conf)

    results: list[torch.Tensor] = []
    for i in range(n):
        s = scores[i].reshape(-1)
        boxes = xyxy[i].reshape(-1, 4)
        labels = classes[i].reshape(-1)
        keep = s > conf_thresh
        boxes, s, labels = boxes[keep], s[keep], labels[keep]
        if boxes.numel() == 0:
            results.append(pred.new_zeros((0, 6)))
            continue
        kept = []
        for c in labels.unique():
            m = labels == c
            idx = nms(boxes[m], s[m], nms_thresh)
            det = torch.cat(
                [boxes[m][idx], s[m][idx, None], labels[m][idx, None].float()],
                dim=1,
            )
            kept.append(det)
        results.append(torch.cat(kept, dim=0) if kept else pred.new_zeros((0, 6)))
    return results
