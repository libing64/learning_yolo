from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv3Loss(nn.Module):
    """YOLOv3 multi-scale loss (Darknet-style).

    Each GT box is assigned to the single best-matching anchor across all scales
    (IoU on width/height). Other anchors in the same cell with IoU > ignore_thresh
    are ignored for objectness.
    """

    def __init__(
        self,
        anchors,
        strides=(8, 16, 32),
        num_classes: int = 80,
        ignore_thresh: float = 0.5,
        lambda_coord: float = 1.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 1.0,
        lambda_cls: float = 1.0,
    ):
        super().__init__()
        self.strides = list(strides)
        self.num_classes = num_classes
        self.ignore_thresh = ignore_thresh
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_cls = lambda_cls
        if not torch.is_tensor(anchors):
            anchors = torch.tensor(anchors, dtype=torch.float32)
        self.register_buffer("anchors", anchors.float().view(len(strides), -1, 2))

    def forward(self, outputs: tuple[torch.Tensor, ...], targets: torch.Tensor, image_size: int):
        device = outputs[0].device
        dtype = outputs[0].dtype
        bs = outputs[0].shape[0]
        na = self.anchors.shape[1]
        n_scales = len(outputs)

        # Flatten anchors for global best-match: (S*A, 2)
        flat_anchors = self.anchors.view(-1, 2)

        loss_xy = torch.zeros((), device=device, dtype=dtype)
        loss_wh = torch.zeros((), device=device, dtype=dtype)
        loss_obj = torch.zeros((), device=device, dtype=dtype)
        loss_noobj = torch.zeros((), device=device, dtype=dtype)
        loss_cls = torch.zeros((), device=device, dtype=dtype)
        n_obj = 0

        # Precompute per-scale tensors
        preds = []
        t_masks, t_xys, t_whs, t_clss, ignores = [], [], [], [], []
        for si, pred in enumerate(outputs):
            n, _, h, w = pred.shape
            pred = pred.view(n, na, 5 + self.num_classes, h, w).permute(0, 1, 3, 4, 2).contiguous()
            preds.append(pred)
            t_masks.append(torch.zeros(n, na, h, w, device=device))
            t_xys.append(torch.zeros(n, na, h, w, 2, device=device, dtype=dtype))
            t_whs.append(torch.zeros(n, na, h, w, 2, device=device, dtype=dtype))
            t_clss.append(torch.zeros(n, na, h, w, self.num_classes, device=device, dtype=dtype))
            ignores.append(torch.zeros(n, na, h, w, device=device, dtype=torch.bool))

        if targets.numel():
            for t in range(targets.shape[0]):
                b = int(targets[t, 0].item())
                cls = int(targets[t, 1].item())
                cx, cy, bw, bh = targets[t, 2:6].tolist()
                wh_pix = torch.tensor([bw * image_size, bh * image_size], device=device, dtype=dtype)

                inter = torch.min(wh_pix, flat_anchors).prod(dim=1)
                union = wh_pix.prod() + flat_anchors.prod(dim=1) - inter
                iou_a = inter / (union + 1e-9)
                best = int(iou_a.argmax().item())
                best_s, best_a = best // na, best % na

                h, w = preds[best_s].shape[2], preds[best_s].shape[3]
                gx, gy = cx * w, cy * h
                gi, gj = min(w - 1, int(gx)), min(h - 1, int(gy))

                # ignore high-IoU anchors on every scale in their projected cell
                for s in range(n_scales):
                    hs, ws = preds[s].shape[2], preds[s].shape[3]
                    gxs, gys = cx * ws, cy * hs
                    gis, gjs = min(ws - 1, int(gxs)), min(hs - 1, int(gys))
                    for a in range(na):
                        idx = s * na + a
                        if idx == best:
                            continue
                        if iou_a[idx] > self.ignore_thresh:
                            ignores[s][b, a, gjs, gis] = True

                t_masks[best_s][b, best_a, gj, gi] = 1
                t_xys[best_s][b, best_a, gj, gi, 0] = gx - gi
                t_xys[best_s][b, best_a, gj, gi, 1] = gy - gj
                t_whs[best_s][b, best_a, gj, gi] = torch.log(
                    (wh_pix / self.anchors[best_s, best_a]).clamp(min=1e-6)
                )
                if 0 <= cls < self.num_classes:
                    t_clss[best_s][b, best_a, gj, gi, cls] = 1.0
                n_obj += 1

        for si, pred in enumerate(preds):
            obj_mask = t_masks[si].bool()
            noobj_mask = (~obj_mask) & (~ignores[si])
            pxy, pwh, pobj, pcls = pred[..., 0:2], pred[..., 2:4], pred[..., 4], pred[..., 5:]

            if obj_mask.any():
                loss_xy = loss_xy + F.binary_cross_entropy_with_logits(
                    pxy[obj_mask], t_xys[si][obj_mask], reduction="sum"
                )
                loss_wh = loss_wh + F.mse_loss(pwh[obj_mask], t_whs[si][obj_mask], reduction="sum")
                loss_obj = loss_obj + F.binary_cross_entropy_with_logits(
                    pobj[obj_mask], torch.ones_like(pobj[obj_mask]), reduction="sum"
                )
                loss_cls = loss_cls + F.binary_cross_entropy_with_logits(
                    pcls[obj_mask], t_clss[si][obj_mask], reduction="sum"
                )
            if noobj_mask.any():
                loss_noobj = loss_noobj + F.binary_cross_entropy_with_logits(
                    pobj[noobj_mask], torch.zeros_like(pobj[noobj_mask]), reduction="sum"
                )

        total = (
            self.lambda_coord * (loss_xy + loss_wh)
            + self.lambda_obj * loss_obj
            + self.lambda_noobj * loss_noobj
            + self.lambda_cls * loss_cls
        ) / bs
        return {
            "loss": total,
            "xy": (loss_xy / bs).detach(),
            "wh": (loss_wh / bs).detach(),
            "obj": (loss_obj / bs).detach(),
            "noobj": (loss_noobj / bs).detach(),
            "cls": (loss_cls / bs).detach(),
            "n_obj": float(n_obj) / max(bs, 1),
        }
