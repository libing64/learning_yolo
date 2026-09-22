from __future__ import annotations

import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)], dim=-1)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU. a: (..., 4), b: (..., 4) broadcastable -> (...)."""
    al, at, ar, ab = a.unbind(-1)
    bl, bt, br, bb = b.unbind(-1)
    inter_w = (torch.min(ar, br) - torch.max(al, bl)).clamp(min=0)
    inter_h = (torch.min(ab, bb) - torch.max(at, bt)).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (ar - al).clamp(min=0) * (ab - at).clamp(min=0)
    area_b = (br - bl).clamp(min=0) * (bb - bt).clamp(min=0)
    return inter / (area_a + area_b - inter + 1e-7)


def make_grid(nx: int, ny: int, device, dtype) -> torch.Tensor:
    gy, gx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=dtype),
        torch.arange(nx, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((gx, gy), dim=-1)  # (ny, nx, 2)


def decode_scale(
    pred: torch.Tensor,
    anchors: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Decode one scale raw prediction to absolute xyxy + obj + cls.

    pred: (N, na*(5+C), H, W)
    returns: (N, na*H*W, 5+C) with xyxy in pixels (relative to input size),
             obj and class logits still raw (caller applies sigmoid).
    """
    n, _, h, w = pred.shape
    na = anchors.shape[0]
    c = pred.shape[1] // na
    pred = pred.view(n, na, c, h, w).permute(0, 1, 3, 4, 2).contiguous()
    grid = make_grid(w, h, pred.device, pred.dtype)
    # YOLOv3 paper: bx = σ(tx)+cx, by = σ(ty)+cy, bw = pw·e^{tw}, bh = ph·e^{th}
    xy = (pred[..., 0:2].sigmoid() + grid) * stride
    wh = pred[..., 2:4].exp() * anchors.view(1, na, 1, 1, 2)
    conf = pred[..., 4:5]
    cls = pred[..., 5:]
    out = torch.cat([xy, wh, conf, cls], dim=-1).view(n, na * h * w, c)
    xyxy = xywh_to_xyxy(out[..., :4])
    return torch.cat([xyxy, out[..., 4:]], dim=-1)


def decode_outputs(
    outputs: tuple[torch.Tensor, ...],
    anchors: torch.Tensor,
    strides: tuple[int, ...],
) -> torch.Tensor:
    """Concatenate decoded predictions from all scales. (N, sum, 5+C)"""
    decoded = []
    for i, pred in enumerate(outputs):
        decoded.append(decode_scale(pred, anchors[i], strides[i]))
    return torch.cat(decoded, dim=1)


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thresh: float = 0.001,
    iou_thresh: float = 0.45,
    max_det: int = 300,
) -> list[torch.Tensor]:
    """NMS on decoded predictions (N, M, 5+C) with raw obj/cls logits.

    Returns list of (D, 6): x1,y1,x2,y2,score,cls  in pixel coords of the network input.
    """
    from torchvision.ops import nms

    n = prediction.shape[0]
    nc = prediction.shape[-1] - 5
    results: list[torch.Tensor] = []
    for i in range(n):
        pred = prediction[i]
        obj = pred[:, 4].sigmoid()
        cls_prob = pred[:, 5:].sigmoid()
        cls_score, cls_id = cls_prob.max(dim=1)
        scores = obj * cls_score
        keep = scores > conf_thresh
        boxes = pred[keep, :4]
        scores = scores[keep]
        cls_id = cls_id[keep]
        if boxes.numel() == 0:
            results.append(prediction.new_zeros((0, 6)))
            continue
        # class-agnostic then per-class nms is standard; do per-class
        kept = []
        for c in cls_id.unique():
            m = cls_id == c
            idx = nms(boxes[m], scores[m], iou_thresh)
            if idx.numel() > max_det:
                idx = idx[:max_det]
            det = torch.cat(
                [boxes[m][idx], scores[m][idx, None], cls_id[m][idx, None].float()],
                dim=1,
            )
            kept.append(det)
        out = torch.cat(kept, dim=0) if kept else prediction.new_zeros((0, 6))
        if out.shape[0] > max_det:
            out = out[out[:, 4].argsort(descending=True)[:max_det]]
        results.append(out)
    return results
